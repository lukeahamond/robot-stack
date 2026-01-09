"""Toy tabletop simulator.

A gripper moves over a unit square. One or two coloured blocks must be carried to
coloured bins named by the instruction. It is deliberately small, but it has the
three properties the architecture is meant to handle:

* multi-step tasks, so there is something to plan;
* slips while carrying, which are recoverable and must be noticed;
* releasing over the wrong bin, which is irreversible and ends the episode.

Swap this class for a ManiSkill or LIBERO adapter to move to a real benchmark. The
rest of the stack only uses: reset, tick, render, proprio, instruction, achieved,
done, success, failed.
"""
import numpy as np

PICK, PLACE, DONE = 0, 1, 2
VERBS = ("pick", "place", "done")
COLORS = ("red", "green", "blue")
PAD = 3  # instruction padding token
PROP = 4  # proprio size: gripper x, y, commanded closed, contact sensor
BIN_XY = np.array([[0.2, 0.85], [0.5, 0.85], [0.8, 0.85]])


def subgoal_text(sg) -> str:
    verb, color = int(sg[0]), int(sg[1])
    if verb == DONE:
        return "done"
    if verb == PICK:
        return f"pick the {COLORS[color]} block"
    return f"place it in the {COLORS[color]} bin"


class ToyKitchen:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        centers = (np.arange(cfg.img) + 0.5) / cfg.img
        self._gx, self._gy = np.meshgrid(centers, centers)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.bin_color = r.permutation(3)                 # colour of each bin slot
        self.n = int(r.integers(1, 3))                    # one or two blocks
        self.block_color = r.permutation(3)[: self.n]
        self.target_bin_color = r.permutation(3)[: self.n]  # distinct target bins
        while True:
            xy = r.uniform([0.12, 0.12], [0.88, 0.5], size=(self.n, 2))
            if self.n == 1 or np.linalg.norm(xy[0] - xy[1]) > 0.25:
                break
        self.block_xy = xy
        self.block_bin = -np.ones(self.n, dtype=int)      # bin slot once placed
        self.grip_xy = r.uniform([0.2, 0.55], [0.8, 0.7])
        self.grip_closed = False
        self.holding = -1
        self.failed = False
        return self

    # ---- task ---------------------------------------------------------------
    @property
    def instruction(self):
        tok = np.full(4, PAD, dtype=np.int64)
        for b in range(self.n):
            tok[2 * b], tok[2 * b + 1] = self.block_color[b], self.target_bin_color[b]
        return tok

    def instruction_text(self) -> str:
        parts = [f"put the {COLORS[self.block_color[b]]} block in the {COLORS[self.target_bin_color[b]]} bin"
                 for b in range(self.n)]
        return ", then ".join(parts)

    @property
    def success(self) -> bool:
        return bool((self.block_bin >= 0).all()) and not self.failed

    @property
    def done(self) -> bool:
        return self.failed or self.success

    def achieved(self):
        """Ground truth for every (verb, colour) subgoal. Used only as a training label."""
        a = np.zeros((2, 3), dtype=bool)
        if self.holding >= 0:
            a[PICK, self.block_color[self.holding]] = True
        for b in range(self.n):
            if self.block_bin[b] >= 0:
                a[PLACE, self.bin_color[self.block_bin[b]]] = True
        return a

    # ---- dynamics (one fast tick) -------------------------------------------
    def tick(self, u, grip_close: bool):
        c = self.cfg
        u = np.clip(np.asarray(u, dtype=float), -c.max_tick, c.max_tick)
        self.grip_xy = np.clip(self.grip_xy + u + self.rng.normal(0, c.tick_noise, 2), 0.0, 1.0)
        self.grip_closed = bool(grip_close)
        if self.holding >= 0:
            self.block_xy[self.holding] = self.grip_xy
            if not grip_close:
                self._release()
            elif self.rng.random() < c.p_slip:
                self._slip()
        elif grip_close:
            free = [b for b in range(self.n) if self.block_bin[b] < 0]
            if free:
                d = [np.linalg.norm(self.block_xy[b] - self.grip_xy) for b in free]
                i = int(np.argmin(d))
                if d[i] < c.grasp_radius:
                    self.holding = free[i]
                    self.block_xy[self.holding] = self.grip_xy

    def _release(self):
        b, self.holding = self.holding, -1
        d = np.linalg.norm(BIN_XY - self.grip_xy, axis=1)
        slot = int(d.argmin())
        if d[slot] < self.cfg.bin_radius:
            self.block_bin[b] = slot
            self.block_xy[b] = BIN_XY[slot]
            if self.bin_color[slot] != self.target_bin_color[b]:
                self.failed = True  # irreversible

    def _slip(self):
        b, self.holding = self.holding, -1
        offset = np.array([self.rng.normal(0, 0.08), -self.rng.uniform(0.12, 0.25)])
        self.block_xy[b] = np.clip(self.grip_xy + offset, [0.1, 0.1], [0.9, 0.6])

    # ---- observations -------------------------------------------------------
    def proprio(self):
        """Gripper x, y, commanded-closed flag and a contact sensor.

        The contact sensor matters: at this camera resolution a held block and a block
        next to a closed gripper look the same, so without it the planner cannot tell
        a grasp from a miss. Real grippers report aperture or force for the same reason.
        """
        return np.array([self.grip_xy[0], self.grip_xy[1], float(self.grip_closed), float(self.holding >= 0)],
                        dtype=np.float32)

    def render(self):
        """Camera image, float32 in [0, 1], shape (3, img, img)."""
        c = self.cfg
        img = np.zeros((3, c.img, c.img), dtype=np.float32)
        for slot in range(3):
            d2 = (self._gx - BIN_XY[slot, 0]) ** 2 + (self._gy - BIN_XY[slot, 1]) ** 2
            img[self.bin_color[slot]] += 0.35 * (d2 < c.bin_radius ** 2)
        for b in range(self.n):
            d2 = (self._gx - self.block_xy[b, 0]) ** 2 + (self._gy - self.block_xy[b, 1]) ** 2
            img[self.block_color[b]] += np.exp(-d2 / (2 * 0.045 ** 2))
        d2 = (self._gx - self.grip_xy[0]) ** 2 + (self._gy - self.grip_xy[1]) ** 2
        img += 0.5 * np.exp(-d2 / (2 * 0.03 ** 2))
        return np.clip(img, 0.0, 1.0)
