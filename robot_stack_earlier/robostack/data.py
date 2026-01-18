"""Episode recording, collection and the training buffer.

Every episode, whether from the expert or from the deployed agent, is stored in one
format so deployment logs can be fed straight back into training.
"""
from collections import defaultdict

import numpy as np
import torch

from .controller import Controller
from .env import DONE, PICK, ToyKitchen
from .expert import expert_action, oracle_subgoal

# EXPERT: demonstrations. SLOPPY: deliberately bad demonstrations, for failure data.
# POLICY: the agent acting alone. CORRECTED: the agent acting, with every step relabelled
# by the expert (the simulator's stand-in for human interventions).
EXPERT, SLOPPY, POLICY, CORRECTED = 0, 1, 2, 3


class Recorder:
    """One episode: T actions and T + 1 frames."""

    def __init__(self, env, mode):
        self.env, self.mode = env, mode
        self.frames, self.prop, self.ach, self.sg = [], [], [], []
        self.action, self.label, self.fail = [], [], []

    def observe(self, subgoal):
        e = self.env
        self.frames.append(np.round(e.render() * 255).astype(np.uint8))
        self.prop.append(e.proprio())
        self.ach.append(e.achieved())
        self.sg.append(np.array(subgoal, dtype=np.int64))

    def act(self, action, label=None):
        """action: what was executed. label: what should be imitated (defaults to the action)."""
        self.action.append(np.asarray(action, dtype=np.float32))
        self.label.append(np.asarray(action if label is None else label, dtype=np.float32))
        self.fail.append(self.env.failed)

    def finish(self):
        T = len(self.action)
        return dict(
            frames=np.stack(self.frames), prop=np.stack(self.prop), achieved=np.stack(self.ach),
            subgoal=np.stack(self.sg),
            action=np.stack(self.action) if T else np.zeros((0, 3), np.float32),
            label=np.stack(self.label) if T else np.zeros((0, 3), np.float32),
            fail=np.array(self.fail, dtype=bool), instr=self.env.instruction,
            success=self.env.success, mode=self.mode,
        )


def collect_episode(cfg, seed, mode=EXPERT):
    """EXPERT episodes are perturbed in two ways, always keeping the expert's intended action
    as the imitation label while a different action is executed:

    * now and then an action is disturbed (waypoint noise, gripper flipped);
    * in some two-block episodes the first pick goes for the other block.

    This puts off-nominal states and their corrections in the data: gripper closed on
    nothing, block dropped, gripper off course, the wrong block in hand. A clean
    demonstrator never visits them, and models trained without them learn shortcuts such
    as "gripper closed means the commanded block is held".
    """
    env = ToyKitchen(cfg, seed)
    rng = np.random.default_rng(seed + 7_000_003)
    ctrl = Controller(cfg)
    sloppy = mode == SLOPPY
    offset = int(rng.integers(1, 3)) if sloppy and rng.random() < 0.5 else 0
    detour = mode == EXPERT and env.n == 2 and rng.random() < cfg.p_detour
    rec = Recorder(env, mode)
    for _ in range(cfg.max_actions):
        a, sg = expert_action(env, rng, noise=0.02 if sloppy else 0.004, slot_offset=offset,
                              p_early_release=0.04 if sloppy else 0.0)
        rec.observe(sg)
        executed = a
        if detour and sg[0] == PICK and (env.block_bin < 0).all():
            executed, _ = expert_action(env, rng, pick_color=int(env.block_color[1]))
        elif mode == EXPERT and rng.random() < cfg.p_perturb:
            executed = a.copy()
            executed[:2] = np.clip(a[:2] + rng.normal(0, 0.04, 2), 0.0, 1.0)
            if rng.random() < 0.5:
                executed[2] = 1.0 - a[2]
        ctrl.track(env, executed[:2], executed[2] > 0.5)
        rec.act(executed, label=a)
        if env.done:
            break
    rec.observe(oracle_subgoal(env))
    return rec.finish()


def seed_base(cfg):
    """First simulator seed used for training data. Each --seed gets its own block of ten million:
    demonstrations from +0, sloppy ones from +500,000, corrections from +1,000,000 and RL episodes
    from +2,000,000. For seed 0 none reach the evaluation seeds (from 9,000,000), and for any other
    seed the block starts above them."""
    if cfg.seed < 0:
        raise ValueError("the seed must be 0 or more")
    return cfg.seed * 10_000_000


def collect(cfg, log=print):
    base = seed_base(cfg)
    eps = [collect_episode(cfg, base + i, EXPERT) for i in range(cfg.n_expert)]
    eps += [collect_episode(cfg, base + 500_000 + i, SLOPPY) for i in range(cfg.n_sloppy)]
    ok = np.mean([e["success"] for e in eps[: cfg.n_expert]])
    bad = np.mean([e["fail"].any() for e in eps[cfg.n_expert:]]) if cfg.n_sloppy else 0.0
    log(f"collected {len(eps)} episodes: expert success {ok:.2f}, sloppy wrong-bin rate {bad:.2f}")
    return eps


class Buffer:
    """Flattens episodes into frame-indexed and step-indexed tensors with derived labels.

    Frame-indexed (one row per frame, T + 1 per episode):
        frames, prop, instr, sg (subgoal label), prev_sg, ach, ret, mode, goal_fi, goal_valid
    Step-indexed (one row per action):
        fi (frame index of the observation), chunk (imitation target), executed, fail, sg, instr, ret, mode, success,
        goal_fi, goal_valid, future_fi
    """

    def __init__(self, cfg, eps=()):
        self.cfg, self.eps, self.Z = cfg, list(eps), None

    def add(self, eps):
        self.eps += list(eps)

    def build(self, device="cpu"):
        c, H = self.cfg, self.cfg.chunk
        F, S = defaultdict(list), defaultdict(list)
        f0 = 0
        for ep in self.eps:
            T, n = len(ep["action"]), len(ep["frames"])
            sg, prop, ach = ep["subgoal"], ep["prop"], ep["achieved"]
            ret = (c.gamma ** (T - np.arange(n))) * float(ep["success"])

            # last step of the subgoal segment each step belongs to
            seg_end = np.zeros(T, dtype=int)
            end = T - 1
            for t in range(T - 1, -1, -1):
                if t < T - 1 and (sg[t] != sg[t + 1]).any():
                    end = t
                seg_end[t] = end
            t_idx = np.arange(T)
            goal_f = seg_end + 1  # frame where the segment's subgoal should have been achieved
            verb, color = sg[:T, 0], sg[:T, 1]
            goal_valid = (verb != DONE) & ach[goal_f, np.minimum(verb, 1), color]

            F["frames"].append(ep["frames"]); F["prop"].append(prop); F["ach"].append(ach)
            F["sg"].append(sg); F["prev_sg"].append(np.concatenate([sg[:1], sg[:-1]]))
            F["instr"].append(np.tile(ep["instr"], (n, 1))); F["ret"].append(ret)
            F["mode"].append(np.full(n, ep["mode"]))
            F["goal_fi"].append(f0 + np.concatenate([goal_f, [T]]))
            F["goal_valid"].append(np.concatenate([goal_valid, [False]]))

            if T:
                # action chunks, relative to the gripper and cut at the subgoal boundary
                chunk = np.zeros((T, H, 3), dtype=np.float32)
                for k in range(H):
                    j = np.minimum(t_idx + k, seg_end)
                    chunk[:, k, :2] = (ep["label"][j, :2] - prop[:T, :2]) / c.max_step
                    chunk[:, k, 2] = ep["label"][j, 2]
                executed = ep["action"].copy()  # what the world model must explain
                executed[:, :2] = (executed[:, :2] - prop[:T, :2]) / c.max_step
                S["fi"].append(f0 + t_idx); S["chunk"].append(chunk); S["fail"].append(ep["fail"])
                S["executed"].append(executed)
                S["sg"].append(sg[:T]); S["instr"].append(np.tile(ep["instr"], (T, 1)))
                S["ret"].append(ret[:T]); S["mode"].append(np.full(T, ep["mode"]))
                S["success"].append(np.full(T, ep["success"]))
                S["goal_fi"].append(f0 + goal_f); S["goal_valid"].append(goal_valid)
                S["future_fi"].append(f0 + np.minimum(t_idx + H, T))
            f0 += n

        def cat(d):
            out = {}
            for k, v in d.items():
                a = np.concatenate(v)
                t = torch.from_numpy(a)
                out[k] = t.float() if a.dtype in (np.float64, np.float32) else t
            return out

        self.F, self.S = cat(F), cat(S)
        self.F = {k: v.to(device) if k != "frames" else v for k, v in self.F.items()}
        self.S = {k: v.to(device) for k, v in self.S.items()}
        self.Z = None
        return self

    @torch.no_grad()
    def encode(self, encoder, device="cpu", bs=1024):
        """Cache the latent of every frame. Everything downstream trains on these."""
        encoder.eval()
        frames = self.F["frames"]
        run_on = next(encoder.parameters()).device
        self.Z = torch.cat([encoder(frames[i:i + bs].to(run_on)).to(device) for i in range(0, len(frames), bs)])
        return self
