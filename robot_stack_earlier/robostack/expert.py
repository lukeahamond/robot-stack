"""Scripted demonstrator. It reads simulator state, so it stands in for a teleoperator."""
import numpy as np

from .env import BIN_XY, DONE, PICK, PLACE


def oracle_subgoal(env):
    """The correct next subgoal, from privileged state. Used as the planner's training label."""
    if env.holding >= 0:
        return (PLACE, int(env.target_bin_color[env.holding]))
    for b in range(env.n):
        if env.block_bin[b] < 0:
            return (PICK, int(env.block_color[b]))
    return (DONE, 0)


def expert_action(env, rng, noise=0.004, slot_offset=0, p_early_release=0.0, pick_color=None):
    """Returns (action, subgoal). action = [waypoint_x, waypoint_y, grip_close].

    slot_offset != 0 sends the block to a wrong bin and p_early_release drops it early.
    Both exist to produce the failure data the critic and world model need.
    pick_color overrides which block a pick goes for (the returned subgoal stays the oracle's).
    """
    c = env.cfg
    verb, color = oracle_subgoal(env)
    subgoal = (verb, color)
    if verb == PICK and pick_color is not None:
        color = pick_color
    p = env.grip_xy.copy()
    if verb == DONE:
        return np.array([p[0], p[1], 0.0], dtype=np.float32), subgoal
    if verb == PICK:
        b = int(np.where(env.block_color == color)[0][0])
        target = env.block_xy[b]
        dist = np.linalg.norm(target - p)
        grip = float(dist < 0.5 * c.grasp_radius)
    else:
        slot = (int(np.where(env.bin_color == color)[0][0]) + slot_offset) % 3
        target = BIN_XY[slot]
        dist = np.linalg.norm(target - p)
        grip = float(dist >= 0.4 * c.bin_radius)
        if rng.random() < p_early_release:
            grip = 0.0
    step = target - p
    if dist > c.max_step:
        step *= c.max_step / dist
    w = np.clip(p + step + rng.normal(0, noise, 2), 0.0, 1.0)
    return np.array([w[0], w[1], grip], dtype=np.float32), subgoal
