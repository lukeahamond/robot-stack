"""Training the world action model on the demonstrations."""
import copy
import math
import time

import torch

from .wam import WAM


def pick_device(cfg):
    if cfg.device != "auto":
        return cfg.device
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def train_wam(cfg, ds, log=print, device=None):
    device = device or pick_device(cfg)
    torch.manual_seed(cfg.seed)
    model = WAM(cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    warm = min(500, cfg.wam_steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / cfg.wam_steps))))
    g = torch.Generator().manual_seed(cfg.seed)
    t0, run_v, run_a = time.time(), 0.0, 0.0
    for step in range(1, cfg.wam_steps + 1):
        idx = ds.train_idx[torch.randint(0, len(ds.train_idx), (cfg.batch,), generator=g)]
        loss, lv, la = model.loss(*ds.batch(idx, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.lerp_(pm, 1 - decay)
        run_v, run_a = 0.98 * run_v + 0.02 * lv, 0.98 * run_a + 0.02 * la
        if step % max(1, cfg.wam_steps // 16) == 0 or step == cfg.wam_steps:
            fix = 1 - 0.98 ** step
            log(f"world action model step {step}/{cfg.wam_steps}: video loss {run_v / fix:.4f}, action loss {run_a / fix:.4f}, "
                f"{time.time() - t0:.0f}s")
        if step % max(1, cfg.wam_steps // 4) == 0 and step < cfg.wam_steps:
            held_out_report(cfg, ema, ds, log)
    return ema.eval()


@torch.no_grad()
def held_out_report(cfg, model, ds, log=print, n=512):
    """How far the generated chunk is from the demonstrator's, on episodes not trained on."""
    device = next(model.parameters()).device
    idx = ds.held_idx[torch.randperm(len(ds.held_idx), generator=torch.Generator().manual_seed(0))[:n]]
    frame, prop, sg, future, chunk = ds.batch(idx, device)
    frames, act = model.sample(frame, prop, sg, generator=torch.Generator().manual_seed(0))
    k = cfg.exec_h
    pos_mm = 1000 * cfg.pos_scale * (act[:, :k, :3] - chunk[:, :k, :3]).norm(dim=-1).mean().item()
    flags = ((act[:, :k, 3:] > 0) == (chunk[:, :k, 3:] > 0)).float().mean((0, 1))
    vid = (frames - model.shrink(future)).abs().mean().item() * 127.5
    log(f"held-out check ({len(idx)} examples): waypoint error {pos_mm:.1f} mm over the executed part, "
        f"gripper {flags[0]:.3f}, touch {flags[1]:.3f}, finished-flag {flags[2]:.3f} correct; "
        f"frame error {vid:.1f} of 255")
    return dict(waypoint_mm=pos_mm, gripper=flags[0].item(), touch=flags[1].item(), finished=flags[2].item(), frame_err=vid)
