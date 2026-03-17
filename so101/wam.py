"""The world action model.

From the camera frames, the body state (joint positions and touch) and a command in words,
one denoising pass produces three things together:

- the next `chunk` frames of joint targets and gripper targets,
- the touch the hand should feel on each of those frames, and whether the command is finished,
- what the cameras will show later, as image features (not pixels).

The cameras are read by a pretrained image encoder (DINOv2). The predicted future is in
that encoder's feature space, taken from a frozen copy. Future prediction can be switched
off at run time; the model is trained with the future tokens hidden part of the time so
that it still works.
"""
import copy
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

HF_NAMES = {"dinov2_small": "facebook/dinov2-small", "dinov2_base": "facebook/dinov2-base"}
N_OUT = 8          # five joints, gripper, expected touch, finished
N_PROP = 7         # six joint positions and touch
KP = 8             # picture coordinates found freely, and again as many steered by the command


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Dino(nn.Module):
    """DINOv2 patch features. Input: (B, 3, H, W) in 0..1. Output: (B, (H/14)*(W/14), dim)."""
    patch = 14

    def __init__(self, name, unfreeze=0, shape=None):
        """`shape` is the encoder's saved configuration. With it the encoder is built empty, for
        weights that are loaded afterwards, and nothing is fetched."""
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if shape is None:
            self.net = AutoModel.from_pretrained(HF_NAMES[name])
        else:
            shape = dict(shape)
            self.net = AutoModel.from_config(AutoConfig.for_model(shape.pop("model_type"), **shape))
        self.dim = self.net.config.hidden_size
        self.requires_grad_(False)
        if unfreeze:
            for block in self.net.encoder.layer[-unfreeze:]:
                block.requires_grad_(True)
            self.net.layernorm.requires_grad_(True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return self.net(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 1:]


class Tiny(nn.Module):
    """A fixed random projection of 14 x 14 patches. Only for checking the plumbing in tests."""
    patch, dim = 14, 48

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("w", torch.randn(self.dim, 3, 14, 14, generator=g) / 14.0)

    def forward(self, x):
        return F.conv2d(x - 0.5, self.w, stride=14).flatten(2).transpose(1, 2)


def make_encoder(name, unfreeze=0, shape=None):
    return Tiny() if name == "tiny" else Dino(name, unfreeze, shape)


def pool(tokens, grid):
    """(B, s*s, D) image tokens averaged down to (B, grid*grid, D)."""
    B, n, D = tokens.shape
    s = int(round(math.sqrt(n)))
    if s == grid:
        return tokens
    x = tokens.transpose(1, 2).reshape(B, D, s, s)
    return F.adaptive_avg_pool2d(x, grid).flatten(2).transpose(1, 2)


class WAM(nn.Module):
    def __init__(self, cfg, cameras, vocab, encoder_shape=None):
        """`cameras` is the list of camera names; `vocab` is the list of words used in commands.

        `encoder_shape` is given when loading a saved model: see `Dino`.
        """
        super().__init__()
        self.cfg, self.cameras, self.vocab = cfg, list(cameras), list(vocab)
        self.word_id = {w: i + 1 for i, w in enumerate(self.vocab)}
        W, C = cfg.width, len(self.cameras)
        self.encoder = make_encoder(cfg.encoder, cfg.unfreeze, encoder_shape)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False) if cfg.unfreeze else None
        D = self.encoder.dim
        self.n_img = C * cfg.grid ** 2
        self.n_fut = len(cfg.future) * C * cfg.fut_grid ** 2
        self.img_in = nn.Linear(D, W)
        # The pretrained encoder says little about colour, so raw patch pixels are fed in beside it.
        self.pixel_in = nn.Conv2d(3, W, self.encoder.patch, self.encoder.patch)
        self.img_pos = nn.Parameter(torch.randn(1, self.n_img, W) * 0.02)
        # A small trainable view of the raw picture at higher resolution. The command chooses what to
        # look for in it, and the answer is a handful of picture coordinates ("where is the thing named").
        self.stem = nn.Sequential(nn.Conv2d(5, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.GELU(),
                                  nn.Conv2d(32, 48, 3, 2, 1), nn.GroupNorm(6, 48), nn.GELU(), nn.Conv2d(48, 48, 3, 1, 1))
        self.free_points = nn.Conv2d(48, KP, 1)
        self.asked_points = nn.Linear(W, KP * 48)
        self.points_in = nn.Linear(C * 2 * KP * 2, W)
        self.prop_in = nn.Linear(N_PROP, W)
        self.word = nn.Embedding(len(self.vocab) + 1, W, padding_idx=0)
        layer = dict(d_model=W, nhead=cfg.heads, dim_feedforward=4 * W, dropout=0.0, activation="gelu",
                     batch_first=True, norm_first=True)
        self.reader = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer), cfg.enc_depth, enable_nested_tensor=False)
        self.denoiser = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer), cfg.dec_depth)
        self.time_in = nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, W))
        self.fut_in, self.fut_out = nn.Linear(D, W), nn.Linear(W, D)
        self.act_in, self.act_out = nn.Linear(N_OUT, W), nn.Linear(W, N_OUT)
        self.out_pos = nn.Parameter(torch.randn(1, self.n_fut + cfg.chunk, W) * 0.02)
        self.norm = nn.LayerNorm(W)
        # Set from the data before training, saved with the weights.
        self.register_buffer("fut_mean", torch.zeros(D))
        self.register_buffer("fut_std", torch.ones(D))

    # ---- inputs -------------------------------------------------------------
    def command_ids(self, commands):
        """Word ids for a list of command strings, padded with 0. Unknown words are dropped."""
        rows = [[self.word_id[w] for w in words(c) if w in self.word_id] or [0] for c in commands]
        out = torch.zeros(len(rows), max(len(r) for r in rows), dtype=torch.long)
        for i, r in enumerate(rows):
            out[i, :len(r)] = torch.tensor(r)
        return out

    def unknown_words(self, command):
        return [w for w in words(command) if w not in self.word_id]

    def context(self, images, prop, cmd):
        """images: (B, C, 3, H, W) in 0..1. prop: (B, 7). cmd: (B, L) word ids."""
        B, C = images.shape[:2]
        flat = images.flatten(0, 1)
        tokens = pool(self.encoder(flat), self.cfg.grid)
        pixels = pool(self.pixel_in(flat - 0.5).flatten(2).transpose(1, 2), self.cfg.grid)
        img = (self.img_in(tokens) + pixels).reshape(B, self.n_img, -1) + self.img_pos
        n = (cmd > 0).sum(1, keepdim=True).clamp(min=1)
        text = self.word(cmd).sum(1) / n.sqrt()
        return self.reader(torch.cat([img, self.points(flat, text)[:, None], self.prop_in(prop)[:, None], text[:, None]], 1))

    def points(self, flat, text):
        """Where things are in each picture, as coordinates. flat: (B*C, 3, H, W). text: (B, W)."""
        n, _, H, Wd = flat.shape
        C = n // len(text)
        ys = torch.linspace(-1, 1, H, device=flat.device).view(1, 1, H, 1).expand(n, 1, H, Wd)
        xs = torch.linspace(-1, 1, Wd, device=flat.device).view(1, 1, 1, Wd).expand(n, 1, H, Wd)
        h = self.stem(torch.cat([flat - 0.5, xs.to(flat.dtype), ys.to(flat.dtype)], 1))          # (n, 48, H/4, W/4)
        query = self.asked_points(text).view(len(text), 1, KP, 48).expand(-1, C, -1, -1).reshape(n, KP, 48)
        maps = torch.cat([self.free_points(h).flatten(2), torch.einsum("nkc,ncp->nkp", query, h.flatten(2))], 1)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, h.shape[2], device=h.device),
                                torch.linspace(-1, 1, h.shape[3], device=h.device), indexing="ij")
        grid = torch.stack([gx.flatten(), gy.flatten()], 1).to(maps.dtype)
        where = torch.softmax(maps.float(), -1).to(maps.dtype) @ grid                               # (n, 2*KP, 2)
        return self.points_in(where.reshape(len(text), -1))

    @torch.no_grad()
    def future_targets(self, frames):
        """Standardised features of later camera frames. frames: (B, n_future, C, 3, H, W)."""
        B = len(frames)
        enc = self.target_encoder or self.encoder
        tokens = pool(enc(frames.flatten(0, 2)), self.cfg.fut_grid)
        return ((tokens - self.fut_mean) / self.fut_std).reshape(B, self.n_fut, -1)

    # ---- denoising ------------------------------------------------------------
    def denoise(self, ctx, fut, act, t, with_future=None):
        """Predicted clean future features and movement, from noisy ones. t: 0 = noise, 1 = clean.

        `with_future` is a (B,) boolean; where False the future tokens are hidden from the
        movement tokens. Pass `fut=None` to leave the future tokens out altogether, which
        gives the same movement as hiding them and costs less.
        """
        W = ctx.shape[-1]
        freqs = torch.exp(-math.log(1000.0) * torch.arange(W // 2, device=ctx.device) / (W // 2))
        ang = t[:, None] * 1000.0 * freqs
        temb = self.time_in(torch.cat([ang.sin(), ang.cos()], -1))[:, None]
        if fut is None:
            x = self.act_in(act) + temb + self.out_pos[:, self.n_fut:]
            return None, self.act_out(self.norm(self.denoiser(x, ctx)))
        x = torch.cat([self.fut_in(fut), self.act_in(act)], 1) + temb + self.out_pos
        mask = None
        if with_future is not None:
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
            mask[:, :self.n_fut] = ~with_future[:, None]
        h = self.norm(self.denoiser(x, ctx, tgt_key_padding_mask=mask))
        return self.fut_out(h[:, :self.n_fut]), self.act_out(h[:, self.n_fut:])

    def loss(self, images, prop, cmd, future_frames, chunk):
        B, dev = len(images), images.device
        fut = self.future_targets(future_frames).float()
        t = torch.sigmoid(torch.randn(B, device=dev) - 0.8)             # mostly high noise
        tv = t[:, None, None]
        keep = torch.rand(B, device=dev) >= self.cfg.p_drop_future
        pf, pa = self.denoise(self.context(images, prop, cmd), tv * fut + (1 - tv) * torch.randn_like(fut),
                              tv * chunk + (1 - tv) * torch.randn_like(chunk), t, keep)
        w = pa.new_tensor([1.0] * 6 + [0.5, 0.5])
        l_act = ((pa.float() - chunk).pow(2) * w).mean()
        per = (pf.float() - fut).pow(2).mean((1, 2))
        l_fut = (per * keep).sum() / keep.sum().clamp(min=1)
        return self.cfg.w_future * l_fut + self.cfg.w_action * l_act, l_fut.item(), l_act.item()

    @torch.no_grad()
    def sample(self, images, prop, cmd, steps=None, with_future=True, generator=None):
        """Movement chunk (B, chunk, 8) and predicted future features (B, n_fut, D) or None."""
        steps = self.cfg.denoise_steps if steps is None else steps
        if steps < 1:                            # with no steps the "movement" would be the starting noise
            raise ValueError(f"denoising needs 1 step or more (it was given {steps!r})")
        B, dev = len(images), images.device
        ctx = self.context(images, prop, cmd)
        act = torch.randn(B, self.cfg.chunk, N_OUT, generator=generator).to(dev)
        fut = torch.randn(B, self.n_fut, self.fut_mean.shape[0], generator=generator).to(dev) if with_future else None
        for i in range(steps):
            t = i / steps
            pf, pa = self.denoise(ctx, fut, act, torch.full((B,), t, device=dev))
            w = (1 / steps) / (1 - t)                # step from the noisy sample towards the predicted clean one
            act = act + w * (pa.float() - act)
            if with_future:
                fut = fut + w * (pf.float() - fut)
        return act, fut
