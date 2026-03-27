"""Demonstrations on disk, and training examples made from them.

`Store` is a flat copy of a recorded dataset: every camera frame already resized, plus
joint positions, commands sent, and touch worked out from the gripper. `lerobot_io.prepare`
makes one from a LeRobot dataset. One recorded episode is one command, and its task string is
the command.

`Examples` turns a store into training batches. For a frame t the model sees the cameras,
joint positions and touch at t, and learns the next `chunk` frames of commands, the touch
on each of those frames, whether the episode is over by then, and what the cameras show
at the `future` offsets.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .controller import GRIP, touch_series
from .wam import N_OUT, words


class StoreWriter:
    def __init__(self, root, cameras, motors, fps, img):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = dict(cameras=list(cameras), motors=list(motors), fps=int(fps), img=int(img), tasks=[], lengths=[])
        self.files = {c: open(self.root / f"frames_{c}.u8", "wb") for c in cameras}
        self.state, self.action = [], []

    def add_episode(self, task, state, action, frames):
        """state, action: (T, 6). frames: {camera: (T, img, img, 3) uint8}."""
        T = len(state)
        for c, f in self.files.items():
            arr = np.ascontiguousarray(frames[c], dtype=np.uint8)
            assert arr.shape == (T, self.meta["img"], self.meta["img"], 3), arr.shape
            f.write(arr.tobytes())
        self.state.append(np.asarray(state, dtype=np.float32))
        self.action.append(np.asarray(action, dtype=np.float32))
        self.meta["tasks"].append(str(task))
        self.meta["lengths"].append(T)

    def close(self):
        for f in self.files.values():
            f.close()
        np.save(self.root / "state.npy", np.concatenate(self.state))
        np.save(self.root / "action.npy", np.concatenate(self.action))
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=1))


class Store:
    def __init__(self, root):
        self.root = Path(root)
        m = json.loads((self.root / "meta.json").read_text())
        self.cameras, self.motors, self.fps, self.img = m["cameras"], m["motors"], m["fps"], m["img"]
        self.tasks = m["tasks"]
        self.lengths = np.array(m["lengths"])
        self.end = np.cumsum(self.lengths)
        self.start = self.end - self.lengths
        self.state = np.load(self.root / "state.npy")
        self.action = np.load(self.root / "action.npy")
        n = len(self.state)
        self.frames = {c: np.memmap(self.root / f"frames_{c}.u8", dtype=np.uint8, mode="r",
                                    shape=(n, self.img, self.img, 3)) for c in self.cameras}
        self.episode = np.repeat(np.arange(len(self.lengths)), self.lengths)

    @property
    def commands(self):
        return sorted(set(self.tasks))


class Norm:
    """Scaling between recorded units and what the model sees. Saved with the weights."""

    def __init__(self, state_mean, state_std, delta_std, grip_mean, grip_std, low, high):
        self.state_mean, self.state_std, self.delta_std = (np.asarray(x, dtype=np.float32) for x in (state_mean, state_std, delta_std))
        self.grip_mean, self.grip_std = float(grip_mean), float(grip_std)
        self.low, self.high = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}

    def prop(self, state, touch):
        """(…, 6) joint positions and (…,) touch 0..1 -> (…, 7)."""
        return np.concatenate([(state - self.state_mean) / self.state_std, 2 * np.asarray(touch)[..., None] - 1], -1).astype(np.float32)

    def targets(self, chunk, state):
        """Model output (H, 8) and the joint positions it was made from -> joint targets (H, 6)."""
        out = np.empty((len(chunk), 6), dtype=np.float64)
        out[:, :GRIP] = state[:GRIP] + chunk[:, :GRIP] * self.delta_std
        out[:, GRIP] = chunk[:, GRIP] * self.grip_std + self.grip_mean
        return out


class Examples:
    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store
        s, c = store, cfg
        self.touch = np.concatenate([
            touch_series(s.state[a:b, GRIP], s.action[a:b, GRIP], c.touch_lag, c.touch_full)
            for a, b in zip(s.start, s.end, strict=True)])
        n = len(s.state)
        last = s.end[s.episode] - 1                                   # last frame of each frame's episode
        steps = np.arange(c.chunk)
        self.ahead = np.minimum(np.arange(n)[:, None] + steps, last[:, None])           # (n, chunk)
        self.finished = (np.arange(n)[:, None] + steps) >= (last[:, None] + 1 - c.tail)
        self.future = np.minimum(np.arange(n)[:, None] + np.array(c.future), last[:, None])
        self.vocab = sorted({w for t in s.tasks for w in words(t)})
        # The model reads a command as the words in it, in any order, so two commands made of the
        # same words would look the same to it.
        same = {}
        for command in s.commands:
            same.setdefault(tuple(sorted(words(command))), []).append(command)
        clash = [group for group in same.values() if len(group) > 1]
        if clash:
            raise ValueError(f"these commands are made of the same words, which the model cannot tell apart: {clash}")
        # Hold out whole episodes, spread across the commands.
        rng = np.random.default_rng(c.seed)
        held = set()
        for task in s.commands:
            mine = rng.permutation([e for e, t in enumerate(s.tasks) if t == task])
            if len(mine) > 1:                                         # never hold out a command's only episode
                held.update(mine[:max(1, int(round(c.val_frac * len(mine))))].tolist())
        if not held:
            raise ValueError("no command has two or more episodes, so none can be held out for the check; "
                             "record at least two episodes of one command")
        is_held = np.array([e in held for e in s.episode])
        self.train_idx = np.flatnonzero(~is_held)
        self.held_idx = np.flatnonzero(is_held)
        self.tail_idx = np.flatnonzero(~is_held & self.finished[:, 0])
        # Scaling comes from the training episodes only. The joint ranges, which become the controller's
        # limits, cover everything recorded.
        tr = self.train_idx
        delta = s.action[self.ahead[tr]][..., :GRIP] - s.state[tr, None, :GRIP]
        self.norm = Norm(s.state[tr].mean(0), s.state[tr].std(0) + 1e-3, delta.reshape(-1, GRIP).std(0) + 1e-3,
                         s.action[tr, GRIP].mean(), s.action[tr, GRIP].std() + 1e-3,
                         np.minimum(s.state.min(0), s.action.min(0)), np.maximum(s.state.max(0), s.action.max(0)))

    def sample(self, rng, n):
        k = int(round(self.cfg.p_tail * n)) if len(self.tail_idx) else 0
        return np.concatenate([rng.choice(self.train_idx, n - k), rng.choice(self.tail_idx, k)]) if k else rng.choice(self.train_idx, n)

    def _frames(self, idx, device):
        """(len(idx), cameras, 3, H, W) floats in 0..1. `idx` may have any shape; it is flattened."""
        flat = np.asarray(idx).reshape(-1)
        order = np.argsort(flat)
        out = []
        for cam in self.store.cameras:
            got = np.empty((len(flat), self.store.img, self.store.img, 3), dtype=np.uint8)
            got[order] = self.store.frames[cam][flat[order]]
            out.append(torch.from_numpy(got))
        x = torch.stack(out, 1).to(device)
        return x.permute(0, 1, 4, 2, 3).float() / 255.0

    def batch(self, idx, device, augment=False, rng=None):
        """images (B, C, 3, H, W), prop (B, 7), commands [str], future frames (B, F, C, 3, H, W), chunk (B, chunk, 8)."""
        s, c, nm = self.store, self.cfg, self.norm
        idx = np.asarray(idx)
        B, nF = len(idx), len(c.future)
        images = self._frames(idx, device)
        future = self._frames(self.future[idx], device).reshape(B, nF, *images.shape[1:])
        if augment:
            images, future = jitter(images, future, max(1, c.shift * s.img // 224) if c.shift else 0, rng)   # shift 0: no shift
        ahead = self.ahead[idx]
        chunk = np.empty((B, c.chunk, N_OUT), dtype=np.float32)
        chunk[..., :GRIP] = (s.action[ahead][..., :GRIP] - s.state[idx][:, None, :GRIP]) / nm.delta_std
        chunk[..., GRIP] = (s.action[ahead][..., GRIP] - nm.grip_mean) / nm.grip_std
        chunk[..., 6] = 2 * self.touch[ahead] - 1
        chunk[..., 7] = np.where(self.finished[idx], 1.0, -1.0)
        prop = nm.prop(s.state[idx], self.touch[idx])
        commands = [s.tasks[e] for e in s.episode[idx]]
        return images, torch.from_numpy(prop).to(device), commands, future, torch.from_numpy(chunk).to(device)


def jitter(images, future, shift, rng):
    """The same small random shift and brightness change for every picture of one example."""
    B = len(images)
    H, W = images.shape[-2:]
    both = torch.cat([images[:, None], future], 1)                    # (B, 1 + F, C, 3, H, W)
    shape = both.shape
    padded = F.pad(both.flatten(1, 3), (shift,) * 4, mode="replicate")
    dx, dy = rng.integers(0, 2 * shift + 1, B), rng.integers(0, 2 * shift + 1, B)
    out = torch.stack([padded[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)]).reshape(shape)
    gain = torch.as_tensor(rng.uniform(0.85, 1.15, B), dtype=out.dtype, device=out.device).view(B, 1, 1, 1, 1, 1)
    out = (out * gain).clamp(0, 1)
    return out[:, 0], out[:, 1:]
