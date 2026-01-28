import numpy as np


class Controller:
    """Fast loop. Tracks one waypoint for `substeps` simulator ticks with a P controller.

    On a real robot this is the whole-body or joint-level controller running at hundreds
    of Hz, usually trained with RL in a physics simulator.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def track(self, env, target_xy, grip_close: bool):
        target_xy = np.asarray(target_xy, dtype=float)
        for _ in range(self.cfg.substeps):
            env.tick(self.cfg.kp * (target_xy - env.grip_xy), grip_close)
            if env.done:
                return
