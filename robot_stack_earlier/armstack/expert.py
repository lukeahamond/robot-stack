"""Scripted demonstrator. It reads the simulator's true block positions, which stands in
for a person teleoperating the arm.

`plan_step` is the whole policy: given the command, the hand position, where the target
is and whether the hand is holding it, it says where the hand should go next, whether the
gripper should be closed, and whether the command is finished. `rollout` imagines that
policy a few steps ahead with no physics; that is the label the world action model learns
to reproduce from pictures.

The approach is a smooth funnel: the hand always heads for the spot above the target and
comes down gradually as it gets close. An earlier version only descended when it was
within 6 mm and backed off otherwise; the camera cannot resolve that, so its
demonstrations looked contradictory and nothing could learn from them.
"""
import numpy as np

from .control import clip_norm
from .sim import DONE, PICK


def funnel(c, ee, xy, z_low):
    """Head for `xy`; height falls from hover to `z_low` as the distance shrinks from 4 cm to 1 cm."""
    d = np.linalg.norm(xy - ee[:2])
    if d > 0.04 and ee[2] < c.z_hover - 0.04:              # far away and low: go straight up first
        return np.array([ee[0], ee[1], c.z_hover]), d
    z = z_low + np.clip((d - 0.01) / 0.03, 0.0, 1.0) * (c.z_hover - z_low)
    return np.array([xy[0], xy[1], z]), d


def plan_step(c, verb, ee, xy, holding):
    """Returns (where to go, gripper closed, command finished)."""
    if verb == PICK:
        if holding:
            if ee[2] >= c.z_lifted:
                return ee, True, True
            return np.array([ee[0], ee[1], c.z_hover]), True, False
        goal, d = funnel(c, ee, xy, c.z_grasp)
        return goal, bool(d < 0.012 and ee[2] < c.z_grasp + 0.008), False
    if not holding:                                         # nothing in the hand to place
        return ee, False, True
    goal, d = funnel(c, ee, xy, c.z_release)
    return goal, not (d < 0.012 and ee[2] < c.z_release + 0.008), False


def rollout(c, verb, ee, xy, holding, grip_now):
    """The next `chunk` steps of the plan: hand offset from now, gripper, expected touch, finished."""
    rows = np.zeros((c.chunk, 6), dtype=np.float32)
    p, grip, finished = np.array(ee, dtype=float), bool(grip_now), False
    for k in range(c.chunk):
        if not finished:
            goal, g, finished = plan_step(c, verb, p, xy, holding)
        if not finished:
            p = p + clip_norm(goal - p, c.max_step)
            if verb == PICK and g and not holding:
                holding = True                              # closing on the block takes one action
            elif verb != PICK and not g:
                holding = False
            grip = g
        rows[k, :3] = (p - ee) / c.pos_scale
        rows[k, 3:] = [1 if grip else -1, 1 if holding else -1, 1 if finished else -1]
    return rows


class Expert:
    def __init__(self, cfg):
        self.cfg = cfg

    def observe(self, env, sg):
        """The few true facts the plan needs: where the target is and whether the hand has it."""
        if sg[0] == PICK:
            i = env.block_of_color(sg[1])
            return env.block_pos(i)[:2], env.held == i
        return env.next_slot(), env.held >= 0

    def act(self, env, subgoal=None):
        c = self.cfg
        sg = env.oracle_subgoal() if subgoal is None else subgoal
        ee = env.ee
        if sg[0] == DONE:
            return ee + clip_norm(np.array([ee[0], ee[1], c.z_hover]) - ee, c.max_step), False, sg
        goal, grip, _ = plan_step(c, sg[0], ee, *self.observe(env, sg))
        return ee + clip_norm(goal - ee, c.max_step), grip, sg
