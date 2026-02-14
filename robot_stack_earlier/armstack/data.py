"""Recording demonstrations and turning them into training examples."""
import multiprocessing as mp

import numpy as np
import torch

from .control import LearnedController, clip_norm
from .expert import Expert, funnel, rollout
from .sim import HELD_TOUCH, PICK
from .sim import ArmEnv

_WORKER = None


def _init(cfg, ctrl_state):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = (cfg, ArmEnv(cfg, LearnedController(cfg, ctrl_state)))


def _episode(seed):
    return run_demo(*_WORKER, seed)


def run_demo(cfg, env, seed):
    """One demonstration, deliberately imperfect in two ways so the data shows recovery:
    some executed actions are nudged off course, and now and then the hand goes for a spot
    a few centimetres from the block and closes on nothing (or on the wrong block). What is
    recorded as the label is never the mistake: it is the demonstrator's plan from the state
    the mistake led to."""
    rng = np.random.default_rng(seed + 1_000_003)
    env.reset(seed, drop=rng.random() < cfg.p_drop)
    expert = Expert(cfg)
    frames, prop, sgs, tgt, holding = [env.frame()], [env.proprio()], [], [], []
    miss, closed, missed_this_pick = None, 0, None
    while env.t < cfg.max_actions and not env.done:
        target, grip, sg = expert.act(env)
        xy, has = expert.observe(env, sg)
        sgs.append(sg); tgt.append(xy); holding.append(has)
        if sg[0] == PICK and not has:
            if miss is None and missed_this_pick != sg and not env.grip_closed and rng.random() < cfg.p_mistake:
                off = rng.normal(size=2)
                miss, closed, missed_this_pick = off / np.linalg.norm(off) * rng.uniform(0.03, 0.09), 0, sg
            if miss is not None:
                goal, d = funnel(cfg, env.ee, xy + miss, cfg.z_grasp)
                grip = bool(d < 0.012 and env.ee[2] < cfg.z_grasp + 0.008)
                target = env.ee + clip_norm(goal - env.ee, cfg.max_step)
                closed += grip
                if closed >= 2:
                    miss = None
        else:
            miss = None
        if rng.random() < cfg.p_perturb:
            target = target.copy()
            target[:2] += rng.normal(0, cfg.perturb_std, 2)
            target[2] = max(target[2] + rng.normal(0, cfg.perturb_std / 3), 0.012)
        env.step(target, grip)
        frames.append(env.frame())
        prop.append(env.proprio())
    return dict(frames=np.stack(frames), prop=np.stack(prop), sg=np.array(sgs, dtype=np.int64),
                tgt=np.array(tgt, dtype=np.float32), holding=np.array(holding), success=env.success, dropped=env.dropped)


def collect(cfg, ctrl_state, log=print, seed0=0):
    global _WORKER
    seeds = range(seed0, seed0 + cfg.n_demos)
    if cfg.workers <= 1:
        # In this process: put back the thread count _init sets for workers, and free the environment.
        threads = torch.get_num_threads()
        _init(cfg, ctrl_state)
        try:
            eps = [_episode(s) for s in seeds]
        finally:
            _WORKER[1].close()
            _WORKER = None
            torch.set_num_threads(threads)
    else:
        with mp.get_context("spawn").Pool(cfg.workers, _init, (cfg, ctrl_state)) as pool:
            eps = list(pool.imap(_episode, seeds, chunksize=4))
    log(f"demonstrations {len(eps)} recorded: success {np.mean([e['success'] for e in eps]):.2f}, "
        f"with a forced drop {np.mean([e['dropped'] for e in eps]):.2f}, "
        f"{sum(len(e['sg']) for e in eps)} actions")
    return eps


def norm_prop(p):
    """Scale hand position, finger opening, gripper command and touch to roughly -1..1."""
    p = np.asarray(p, dtype=np.float32)
    return np.stack([(p[..., 0] - 0.36) / 0.15, p[..., 1] / 0.25, (p[..., 2] - 0.08) / 0.08,
                     p[..., 3] / 0.045 - 1, p[..., 4] * 2 - 1, p[..., 5] * 2 - 1], -1)


class Dataset:
    """Each example: the frame and body state at one moment, the command in force, the
    demonstrator's plan for the next `chunk` actions from exactly that state (hand waypoints
    relative to the hand now, gripper, expected touch, and a flag for "this command is
    finished"), and the two camera frames that followed.

    One extra example per command shows the state just after it ended, labelled finished.
    """

    def __init__(self, cfg, eps):
        H = cfg.chunk
        frames, prop, fi, chunk, fut, sg_rows, ep_id = [], [], [], [], [], [], []
        base = 0
        good = [e for e in eps if e["success"]]
        if len(good) < 2:
            raise ValueError(f"only {len(good)} of {len(eps)} demonstrations succeeded; at least 2 are needed, "
                             f"one to train on and one to hold out. Record more (n_demos).")
        for n, ep in enumerate(good):
            T = len(ep["sg"])
            sg, p = ep["sg"], ep["prop"]
            end = np.zeros(T, dtype=int)
            for t in range(T - 1, -1, -1):
                end[t] = t if t == T - 1 or (sg[t] != sg[t + 1]).any() else end[t + 1]
            for t in range(T):
                fi.append(base + t); sg_rows.append(sg[t]); ep_id.append(n)
                chunk.append(rollout(cfg, sg[t, 0], p[t, :3], ep["tgt"][t], bool(ep["holding"][t]), p[t, 4] > 0.5))
                fut.append([base + min(t + f, end[t] + 1) for f in cfg.future])
            for e in np.unique(end):                       # the state just after each command ended
                row = np.zeros((H, 6), dtype=np.float32)
                row[:, 3], row[:, 4], row[:, 5] = p[e + 1, 4] * 2 - 1, np.where(p[e + 1, 5] > HELD_TOUCH, 1.0, -1.0), 1.0
                fi.append(base + e + 1); chunk.append(row); sg_rows.append(sg[e]); ep_id.append(n)
                fut.append([base + e + 1] * len(cfg.future))
            frames.append(ep["frames"]); prop.append(ep["prop"])
            base += T + 1
        self.frames = torch.from_numpy(np.concatenate(frames))
        self.prop = torch.from_numpy(norm_prop(np.concatenate(prop)))
        self.fi = torch.tensor(fi)
        self.chunk = torch.from_numpy(np.stack(chunk))
        self.fut = torch.tensor(fut)
        self.sg = torch.from_numpy(np.stack(sg_rows))
        ep_id = np.array(ep_id)
        n_eps = ep_id.max() + 1
        held = ep_id >= min(int(0.95 * n_eps), n_eps - 1)      # at least one episode held out, at least one trained on
        self.train_idx = torch.from_numpy(np.where(~held)[0])
        self.held_idx = torch.from_numpy(np.where(held)[0])

    def batch(self, idx, device):
        fi = self.fi[idx]
        to = lambda x: x.to(device)
        return (to(self.frames[fi]).float() / 127.5 - 1, to(self.prop[fi]), to(self.sg[idx]),
                to(self.frames[self.fut[idx]]).float() / 127.5 - 1, to(self.chunk[idx]))
