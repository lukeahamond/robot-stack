"""The fast layer. It runs at 50 Hz and turns "put the fingertip here" into joint targets.

`LearnedController` is a small network trained on simulated arm motions to do what
`AnalyticController` does with trigonometry: given the joint angles and a small error at
the fingertip, say how far to turn each joint. Because it is fed the measured error on
every tick, its own mistakes are corrected on the next tick.
"""
import numpy as np
import torch
import torch.nn as nn

from .sim import fk, ik


def clip_norm(v, limit):
    n = np.linalg.norm(v)
    return v if n <= limit else v * (limit / n)


class AnalyticController:
    def __init__(self, cfg):
        self.cfg = cfg

    def tick(self, u, ee, target):
        err = clip_norm(target - ee, self.cfg.ctrl_clip)
        return ik(fk(u) + err)


def _features(u, err, clip):
    return np.concatenate([np.sin(u[..., :1]), np.cos(u[..., :1]), u[..., 1:], err / clip], -1)


OUT_SCALE = 0.1


def controller_net(cfg):
    h = cfg.ctrl_hidden
    return nn.Sequential(nn.Linear(7, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))


class LearnedController:
    def __init__(self, cfg, state):
        self.cfg = cfg
        self.net = controller_net(cfg)
        self.net.load_state_dict(state)
        self.net.eval()

    @torch.no_grad()
    def tick(self, u, ee, target):
        err = clip_norm(target - ee, self.cfg.ctrl_clip)
        x = torch.as_tensor(_features(u, err, self.cfg.ctrl_clip), dtype=torch.float32)
        return u + OUT_SCALE * self.net(x).numpy().astype(np.float64)


def train_controller(cfg, log=print):
    """Supervised: sample arm poses and small fingertip errors, label with the exact joint change."""
    rng = np.random.default_rng(cfg.seed)
    # Only poses the arm can reach: in the corners of the box `ik` gives up, and the label would be wrong.
    pos = np.empty((0, 3))
    while len(pos) < cfg.ctrl_samples:
        p = rng.uniform([0.18, -0.36, 0.0], [0.56, 0.36, 0.24], size=(cfg.ctrl_samples, 3))
        q = ik(p)
        pos = np.concatenate([pos, p[(np.linalg.norm(fk(q) - p, axis=1) < 1e-6) & (q[:, 2] >= 0.1)]])   # 0.1: elbow limit
    pos = pos[:cfg.ctrl_samples]
    u = ik(pos)
    err = rng.normal(size=pos.shape)
    err *= rng.uniform(0, cfg.ctrl_clip, size=(len(pos), 1)) / np.linalg.norm(err, axis=1, keepdims=True)
    x = torch.tensor(_features(u, err, cfg.ctrl_clip), dtype=torch.float32)
    y = torch.tensor((ik(fk(u) + err) - u) / OUT_SCALE, dtype=torch.float32)
    torch.manual_seed(cfg.seed)
    net = controller_net(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.ctrl_steps)
    g = torch.Generator().manual_seed(cfg.seed)
    n_train = int(0.95 * len(x))
    for _ in range(cfg.ctrl_steps):
        idx = torch.randint(0, n_train, (1024,), generator=g)
        loss = (net(x[idx]) - y[idx]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        held = (net(x[n_train:]) - y[n_train:]).pow(2).mean().sqrt().item()
        scale = y[n_train:].pow(2).mean().sqrt().item()
    log(f"controller   held-out error {100 * held / scale:.1f}% of the typical joint step")
    return net.state_dict()
