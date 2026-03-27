"""Training the world action model, the held-out check, and saving and loading it."""
import copy
import math
import queue
import threading
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from .config import from_saved
from .controller import GRIP
from .data import Norm
from .wam import WAM, pool


def pick_device(cfg):
    if cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train(cfg, ex, log=print, device=None, init=None):
    """`ex` is an `Examples`. Returns the trained model (the running average of the weights).

    `init` is a saved model to continue from instead of starting fresh.
    """
    device = device or pick_device(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = WAM(cfg, ex.store.cameras, ex.vocab).to(device)
    if init:
        # A saved model leaves out the frozen copy of the encoder; anything else missing is a mismatch.
        saved = torch.load(init, map_location=device, weights_only=True)
        # The weights are tied to the order of the words and cameras, which the tensor shapes do not show.
        for what, old, new in (("words", saved.get("vocab"), model.vocab), ("cameras", saved.get("cameras"), model.cameras)):
            if old is not None and list(old) != new:
                raise ValueError(f"{init} does not fit this data: it was trained with {what} {list(old)}, this data has {new}")
        try:
            found = model.load_state_dict(saved["state"], strict=False)
        except RuntimeError as e:                       # tensors of other shapes: trained with other model settings
            raise ValueError(f"{init} does not fit this model: it was trained with other model settings "
                             f"(see TRAINED in so101/config.py), so its weights have other shapes") from e
        missing = [k for k in found.missing_keys if not k.startswith("target_encoder.")]
        if missing or found.unexpected_keys:
            raise ValueError(f"{init} does not fit this model: missing {missing}, unexpected {list(found.unexpected_keys)}")
        # The weights were trained against the saved scaling. Scaling from this data instead would shift
        # every movement the model produces. Only the joint ranges widen to cover the new data.
        old = Norm(**saved["norm"])
        old.low, old.high = np.minimum(old.low, ex.norm.low), np.maximum(old.high, ex.norm.high)
        ex.norm = old
        log(f"scaling kept from {init}")
    if not init:
        # A continued model keeps the scale its future head was trained against.
        set_future_scale(model, ex, device, rng)
    ema = copy.deepcopy(model).requires_grad_(False)
    live = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    averaged = dict(ema.named_parameters())
    groups = [dict(params=[p for n, p in live if n.startswith("encoder.")], lr=cfg.enc_lr),
              dict(params=[p for n, p in live if not n.startswith("encoder.")], lr=cfg.lr)]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / cfg.steps))))
    amp = cfg.amp and str(device).startswith("cuda")     # bfloat16 has float32's range, so no loss scaling
    n_live = sum(p.numel() for _, p in live)
    log(f"world action model: {n_live / 1e6:.1f}M parameters being trained, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M in total, on {device}")
    t0, run_f, run_a = time.time(), 0.0, 0.0
    make = lambda: ex.batch(ex.sample(rng, cfg.batch), device, augment=True, rng=rng)  # noqa: E731
    for step, (images, prop, commands, future, chunk) in enumerate(prefetched(make, cfg.steps, cfg.prefetch), 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, lf, la = model.loss(images, prop, model.command_ids(commands).to(device), future, chunk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in live], 1.0)
        opt.step()
        sched.step()
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for n, p in live:
                averaged[n].lerp_(p, 1 - decay)
        run_f, run_a = 0.98 * run_f + 0.02 * lf, 0.98 * run_a + 0.02 * la
        if step % max(1, cfg.steps // 20) == 0 or step == cfg.steps:
            fix = 1 - 0.98 ** step
            log(f"step {step}/{cfg.steps}: movement loss {run_a / fix:.4f}, future loss {run_f / fix:.4f}, {time.time() - t0:.0f}s")
        if step % max(1, cfg.steps // 4) == 0 and step < cfg.steps:
            held_out_check(cfg, ema, ex, log)
    return ema.eval()


def prefetched(make, n, ahead):
    """`make()` called `n` times, on another thread up to `ahead` calls in front, so reading frames from
    disk overlaps the training step. The calls run one after another in order, so the batches (and the
    random numbers they use) are the same as without prefetching. `ahead` 0 makes them here, in turn."""
    if ahead <= 0:
        for _ in range(n):
            yield make()
        return
    out, stop = queue.Queue(maxsize=ahead), threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                out.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def work():
        try:
            for _ in range(n):
                if not put(make()):
                    return
        except BaseException as e:                     # surfaced on the training thread
            put(e)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        for _ in range(n):
            item = out.get()
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join()


@torch.no_grad()
def set_future_scale(model, ex, device, rng, n=128):
    """Mean and spread of the image features the model is asked to predict."""
    images = ex._frames(rng.choice(ex.train_idx, min(n, len(ex.train_idx)), replace=False), device)
    enc = model.target_encoder or model.encoder
    tokens = pool(enc(images.flatten(0, 1)), ex.cfg.fut_grid).flatten(0, 1)
    model.fut_mean.copy_(tokens.mean(0))
    model.fut_std.copy_(tokens.std(0) + 1e-3)


@torch.no_grad()
def held_out_check(cfg, model, ex, log=print, n=256):
    """How close the generated chunk is to the recorded one, on episodes not trained on."""
    device = next(model.parameters()).device
    rng = np.random.default_rng(0)
    idx = rng.choice(ex.held_idx, min(n, len(ex.held_idx)), replace=False)
    images, prop, commands, future, chunk = ex.batch(idx, device)
    cmd = model.command_ids(commands).to(device)
    gen = torch.Generator().manual_seed(0)
    act, fut = model.sample(images, prop, cmd, generator=gen)
    act_only, _ = model.sample(images, prop, cmd, with_future=False, generator=torch.Generator().manual_seed(0))
    k = cfg.execute
    nm = ex.norm
    scale = torch.as_tensor(nm.delta_std, device=device)

    def joint_error(a):
        return ((a[:, :k, :GRIP] - chunk[:, :k, :GRIP]).abs() * scale).mean().item()

    grip = ((act[:, :k, GRIP] - chunk[:, :k, GRIP]).abs() * nm.grip_std).mean().item()
    flags = ((act[:, :k, 6:] > 0) == (chunk[:, :k, 6:] > 0)).float().mean((0, 1))
    target = model.future_targets(future).float()
    now = model.future_targets(images[:, None].expand(-1, len(cfg.future), -1, -1, -1, -1)).float()
    out = dict(joint_error=joint_error(act), joint_error_no_future=joint_error(act_only), gripper_error=grip,
               touch_correct=flags[0].item(), finished_correct=flags[1].item(),
               future_match=F.cosine_similarity(fut, target, dim=-1).mean().item(),
               future_match_if_nothing_moved=F.cosine_similarity(now, target, dim=-1).mean().item())
    log(f"held-out check ({len(idx)} examples, first {k} frames of each chunk): joint error {out['joint_error']:.2f} units "
        f"({out['joint_error_no_future']:.2f} with future prediction off), gripper error {grip:.2f}, "
        f"touch {flags[0]:.3f} and finished flag {flags[1]:.3f} correct; predicted future matches the real one "
        f"{out['future_match']:.3f} (copying the present would score {out['future_match_if_nothing_moved']:.3f})")
    return out


def save(path, cfg, model, ex):
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("target_encoder.")}
    # The encoder's configuration is saved too, so loading builds it without fetching the pretrained one.
    net = getattr(model.encoder, "net", None)
    shape = None if net is None else net.config.to_diff_dict()
    torch.save(dict(cfg=asdict(cfg), cameras=model.cameras, vocab=model.vocab, norm=ex.norm.to_dict(),
                    fps=ex.store.fps, motors=ex.store.motors, commands=ex.store.commands, state=state,
                    encoder_shape=shape), path)


def load(path, device="cpu", settings=None):
    """Returns (cfg, model, info) where info has norm, fps, motors and the commands trained on.

    `settings` is a `Config` whose run-time settings (controller, events, planner) are used
    in place of the ones saved with the model. See `config.from_saved`.
    """
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = from_saved(ck["cfg"], settings)
    model = WAM(cfg, ck["cameras"], ck["vocab"], encoder_shape=ck.get("encoder_shape"))   # older files have none
    model.target_encoder = None
    model.load_state_dict(ck["state"])
    info = dict(norm=Norm(**ck["norm"]), fps=ck["fps"], motors=ck["motors"], commands=ck["commands"], cameras=ck["cameras"])
    return cfg, model.to(device).eval(), info
