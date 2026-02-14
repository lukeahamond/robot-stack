"""A small world action model.

One transformer takes the current camera frame, the body state (including touch) and the
command, and generates two things together in a single denoising pass: the next camera
frames, and the next chunk of actions with the touch it expects to feel. It is trained as
a denoiser: noise is mixed into the future frames and the action chunk, and the model
learns to output the clean versions. (It predicts the clean data, not the noise, because
a frame patch has more numbers than the model is wide, so noise cannot pass through it.)
"""
import math

import torch
import torch.nn as nn


class WAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, self.g, self.p = cfg.width, cfg.img // cfg.patch, cfg.patch
        self.gf = cfg.img // cfg.fut_down // cfg.patch          # generated frames are smaller than the input frame
        self.n_ctx, self.n_patch = self.g * self.g, self.gf * self.gf
        self.patch_dim, self.n_fut = cfg.patch * cfg.patch * 3, len(cfg.future)
        # current frame: a small convolutional stem with coordinate channels, one token per patch
        self.stem = nn.Sequential(nn.Conv2d(5, 48, 3, 2, 1), nn.GroupNorm(8, 48), nn.GELU(),
                                  nn.Conv2d(48, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),
                                  nn.Conv2d(96, 96, 3, 1, 1), nn.GroupNorm(8, 96), nn.GELU())
        self.to_tokens = nn.Conv2d(96, d, cfg.patch // 4, cfg.patch // 4)
        # keypoints: a feature map reduced to the image position where it is strongest, which gives
        # precise locations that coarse patch tokens are slow to learn. Eight are the same for every
        # command; eight are steered by the command, so "red" can ask where the red thing is.
        self.keypoints = nn.Conv2d(96, 8, 1)
        self.query = nn.Linear(d, 8 * 96)
        self.kp_in = nn.Sequential(nn.Linear(32, d), nn.GELU(), nn.Linear(d, d))
        q = cfg.img // 4
        ky, kx = torch.meshgrid(torch.linspace(-1, 1, q), torch.linspace(-1, 1, q), indexing="ij")
        self.register_buffer("kp_grid", torch.stack([kx, ky], -1).reshape(-1, 2), persistent=False)
        self.prop_in = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.verb, self.color = nn.Embedding(2, d), nn.Embedding(4, d)
        self.vid_in, self.act_in = nn.Linear(self.patch_dim, d), nn.Linear(6, d)
        self.time_in = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        n_tokens = self.n_ctx + 3 + self.n_fut * self.n_patch + cfg.chunk
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.vid_out, self.act_out = nn.Linear(d, self.patch_dim), nn.Linear(d, 6)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, cfg.img), torch.linspace(-1, 1, cfg.img), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None], persistent=False)

    # frames are (B, H, W, 3) in -1..1; patches are (B, n_patch, patch_dim)
    def patchify(self, x):
        B, g, p = x.shape[0], self.gf, self.p
        return x.reshape(B, g, p, g, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, p * p * 3)

    def unpatchify(self, x):
        B, g, p = x.shape[0], self.gf, self.p
        return x.reshape(B, g, g, p, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(B, g * p, g * p, 3)

    def context(self, frame, prop, sg):
        x = torch.cat([frame.permute(0, 3, 1, 2), self.coords.expand(len(frame), -1, -1, -1)], 1)
        h = self.stem(x)
        img = self.to_tokens(h).flatten(2).transpose(1, 2)
        cmd = self.verb(sg[:, 0]) + self.color(sg[:, 1])
        asked = torch.einsum("bkc,bcn->bkn", self.query(cmd).view(-1, 8, 96), h.flatten(2))
        where = torch.softmax(torch.cat([self.keypoints(h).flatten(2), asked], 1), -1) @ self.kp_grid   # (B, 16, 2)
        return torch.cat([img, self.kp_in(where.flatten(1))[:, None], self.prop_in(prop)[:, None], cmd[:, None]], 1)

    def denoise(self, ctx, vid, act, t):
        """Predicted clean future-frame patches and action chunk, from noisy ones at noise level 1 - t."""
        d = ctx.shape[-1]
        freqs = torch.exp(-math.log(1000.0) * torch.arange(d // 2, device=ctx.device) / (d // 2))
        ang = t[:, None] * 1000.0 * freqs
        temb = self.time_in(torch.cat([ang.sin(), ang.cos()], -1))[:, None]
        B = len(ctx)
        v = self.vid_in(vid.reshape(B, -1, self.patch_dim)) + temb
        a = self.act_in(act) + temb
        h = self.norm(self.blocks(torch.cat([ctx, v, a], 1) + self.pos))
        n = ctx.shape[1]
        return self.vid_out(h[:, n:n + v.shape[1]]), self.act_out(h[:, n + v.shape[1]:])

    def shrink(self, frames):
        """Average frames (..., H, W, 3) down to the size the model generates."""
        k = self.cfg.fut_down
        s = frames.shape
        return frames.reshape(*s[:-3], s[-3] // k, k, s[-2] // k, k, 3).mean((-4, -2))

    def loss(self, frame, prop, sg, future, chunk):
        B = len(frame)
        future = self.shrink(future)
        vid = torch.cat([self.patchify(future[:, i]) for i in range(self.n_fut)], 1)
        t = torch.sigmoid(torch.randn(B, device=frame.device) - 0.8)      # mostly high noise
        nv, na = torch.randn_like(vid), torch.randn_like(chunk)
        tv = t[:, None, None]
        pv, pa = self.denoise(self.context(frame, prop, sg), tv * vid + (1 - tv) * nv, tv * chunk + (1 - tv) * na, t)
        w = pa.new_tensor([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])     # positions are small numbers next to the on/off flags
        l_vid, l_act = (pv - vid).pow(2).mean(), ((pa - chunk).pow(2) * w).mean()
        return l_vid + self.cfg.w_action * l_act, l_vid.item(), l_act.item()

    @torch.no_grad()
    def sample(self, frame, prop, sg, steps=None, generator=None):
        """Generate future frames (B, n_fut, h, w, 3) and an action chunk (B, chunk, 6) together."""
        steps = steps or self.cfg.denoise_steps
        B, dev = len(frame), frame.device
        ctx = self.context(frame, prop, sg)
        vid = torch.randn(B, self.n_fut * self.n_patch, self.patch_dim, generator=generator).to(dev)
        act = torch.randn(B, self.cfg.chunk, 6, generator=generator).to(dev)
        for i in range(steps):
            t = i / steps
            pv, pa = self.denoise(ctx, vid, act, torch.full((B,), t, device=dev))
            w = (1 / steps) / (1 - t)                 # step from the noisy sample towards the predicted clean one
            vid, act = vid + w * (pv - vid), act + w * (pa - act)
        frames = torch.stack([self.unpatchify(vid[:, i * self.n_patch:(i + 1) * self.n_patch]) for i in range(self.n_fut)], 1)
        return frames.clamp(-1, 1), act
