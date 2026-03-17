"""The fast layer: what runs on every tick between calls to the world action model.

On the SO-101 the servos close their own position loop, so nothing is learned here. This
layer turns a chunk of joint targets at the dataset's frame rate into smooth targets at
the tick rate, enforces limits, protects the gripper, and measures touch.

Touch on this arm has no dedicated sensor. It is read from the gripper: if the gripper
stays further open than every command it was recently given, something is in the way.
The same rule is applied to recorded demonstrations and to the live arm.
"""
from collections import deque

import numpy as np

GRIP = 5        # index of the gripper in the six joint values


def touch_from_gap(grip_state, recent_commands, full):
    """0..1. `recent_commands` are the gripper targets sent before this reading."""
    if len(recent_commands) == 0:
        return 0.0
    return float(np.clip((grip_state - max(recent_commands)) / full, 0.0, 1.0))


def touch_series(grip_state, grip_action, lag, full):
    """Touch for every frame of a recorded episode, by the same rule."""
    T = len(grip_state)
    out = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        out[t] = touch_from_gap(grip_state[t], grip_action[max(0, t - lag):t], full)
    if T > 1:
        out[0] = out[1]            # nothing was sent before the first frame; assume touch did not just change
    return out


class TouchSensor:
    def __init__(self, cfg):
        self.full = cfg.touch_full
        self.sent = deque(maxlen=cfg.touch_lag * cfg.ticks_per_frame)

    def read(self, grip_state):
        return touch_from_gap(grip_state, self.sent, self.full)

    def sent_command(self, grip_target):
        self.sent.append(float(grip_target))


class Plan:
    """One chunk from the world action model, anchored at the tick its observation was taken."""

    def __init__(self, tick0, targets, touch, finished, ticks_per_frame):
        self.tick0, self.targets, self.touch, self.finished = tick0, targets, touch, finished
        self.tpf = ticks_per_frame

    def frame(self, tick):
        return (tick - self.tick0) / self.tpf

    def target(self, tick):
        """Joint targets at `tick`, interpolated between frames and held after the last one."""
        f = float(np.clip(self.frame(tick), 0.0, len(self.targets) - 1))
        i = int(f)
        j = min(i + 1, len(self.targets) - 1)
        return (1 - (f - i)) * self.targets[i] + (f - i) * self.targets[j]

    def expected_touch(self, tick):
        return float(self.touch[int(np.clip(round(self.frame(tick)), 0, len(self.touch) - 1))])


class Blend:
    """Targets moving from one trajectory to another over `span` ticks from `start`."""

    def __init__(self, old, new, start, span):
        self.old, self.new, self.start, self.span = old, new, start, span

    def share(self, tick):
        return (tick - self.start + 1) / self.span if self.span else 1.0

    def target(self, tick):
        a = self.share(tick)
        new = self.new.target(tick)
        return new if a >= 1.0 else (1 - a) * self.old.target(tick) + a * new

    def expected_touch(self, tick):
        """The touch expected by whichever trajectory has the larger share of the targets now."""
        return (self.new if self.share(tick) >= 0.5 else self.old).expected_touch(tick)


class Controller:
    def __init__(self, cfg, fps, low, high):
        """`low` and `high` are the joint ranges seen in the demonstrations."""
        self.cfg = cfg
        self.dt = 1.0 / (fps * cfg.ticks_per_frame)
        self.low = np.asarray(low, dtype=float) - cfg.limit_margin
        self.high = np.asarray(high, dtype=float) + cfg.limit_margin
        self.low[GRIP], self.high[GRIP] = low[GRIP], high[GRIP]
        self.plan = self.old = None
        self.blend_from = 0
        self.last = None
        self._free_all()

    def _free_all(self):
        """Forget which joints are blocked. The watch starts again from the next tick."""
        self.blocked = np.zeros(GRIP, dtype=bool)
        self.anchor = self.since = None

    def _leads(self, tick, state, ask):
        """How far each joint may be aimed from where it is: `max_lead`, or `idle_lead` while it is blocked.

        `ask` is how far the targets are from where the joints are. A joint is watched while `ask` is more
        than `idle_lead`. If in `block_time` it has not moved `idle_lead` towards its target, it is blocked.
        """
        c, pos = self.cfg, state[:GRIP]
        if self.anchor is None:
            self.anchor, self.since = pos.copy(), np.full(GRIP, tick)
        far = np.abs(ask) > c.idle_lead
        progress = (pos - self.anchor) * np.sign(ask) >= c.idle_lead
        restart = ~far | progress                         # not asked far, or following: watch again from here
        self.anchor[restart], self.since[restart], self.blocked[restart] = pos[restart], tick, False
        self.blocked |= far & (tick - self.since >= max(1, round(c.block_time / self.dt)))
        return np.where(self.blocked, c.idle_lead, c.max_lead)

    def _blend(self):
        return Blend(self.old, self.plan, self.blend_from, self.cfg.blend * self.cfg.ticks_per_frame)

    def reset(self, state):
        """Start a new command from where the arm is.

        The gripper keeps its last commanded value if there is one. Where it is and where it
        was told to go differ while it squeezes an object, and starting again from where it
        is would let go of the squeeze.
        """
        self.plan = self.old = None
        self._free_all()
        grip = None if self.last is None else self.last[GRIP]
        self.last = np.asarray(state, dtype=float).copy()
        if grip is not None:
            self.last[GRIP] = grip

    def hold(self):
        """Stop following any plan and stay at the last commanded position."""
        self.plan = self.old = None

    def set_plan(self, plan, tick):
        """The new plan takes over from the targets being followed now. If a blend is still under way,
        that blend is what it takes over from, not the plan that was blending in."""
        if self.plan is not None and self.old is not None and self._blend().share(tick) < 1.0:
            old = self._blend()
        else:
            old = self.plan
        self.old, self.plan, self.blend_from = old, plan, tick

    def expected_touch(self, tick):
        """The touch expected by the targets being followed at `tick`, or None while there is no plan."""
        if self.plan is None:
            return None
        return (self.plan if self.old is None else self._blend()).expected_touch(tick)

    def tick(self, tick, state):
        c = self.cfg
        if self.plan is None:
            want = self.last.copy()
        elif self.old is None:
            want = self.plan.target(tick)
        else:
            want = self._blend().target(tick)
        # The range comes first, so the speed limit below also covers the way back into it. An arm
        # that starts outside the range seen in the data is brought in at the speed limit, not at once.
        want = np.clip(want, self.low, self.high)
        # Do not run away from a blocked joint: aim no further than `max_lead` from where it is. The speed
        # limit comes last, so if the arm is pushed far from its target the target follows at the speed
        # limit instead of jumping. A joint that stays blocked is aimed only `idle_lead` ahead: the speed limit is
        # on the target, not on the joint, so letting go of a joint aimed `max_lead` ahead would snap it that far
        # at the servo's own speed. Coming down from `max_lead` to `idle_lead` also goes through the speed limit.
        ask = want[:GRIP] - state[:GRIP]
        lead = self._leads(tick, state, ask)
        want[:GRIP] = state[:GRIP] + np.clip(ask, -lead, lead)
        step = c.max_speed * self.dt
        want[:GRIP] = self.last[:GRIP] + np.clip(want[:GRIP] - self.last[:GRIP], -step, step)
        # The gripper has no speed limit, on purpose: it has to close as fast as the leader did in the
        # demonstrations to grasp, and the squeeze limit below already keeps it from stalling. The one
        # exception is a gripper target outside the range seen in the data (the arm started there): it is
        # brought into the range at the speed limit, as the joints are, not at once.
        # The squeeze limit comes after the range, so a gripper reading above the range is never squeezed
        # past the limit by the range pulling its target down. Above the range the target then only
        # follows the gripper to where it already is.
        want[GRIP] = np.clip(want[GRIP], self.low[GRIP], self.high[GRIP])
        if not self.low[GRIP] <= self.last[GRIP] <= self.high[GRIP]:
            want[GRIP] = self.last[GRIP] + np.clip(want[GRIP] - self.last[GRIP], -step, step)
        want[GRIP] = max(want[GRIP], state[GRIP] - c.grip_squeeze)   # do not stall the gripper motor
        # NaN passes through every limit above. Refuse it before it becomes the target the arm holds, so
        # `last` stays the last good target and easing the grip afterwards still sends real numbers.
        if not np.isfinite(want).all():
            raise ValueError(f"the controller was given targets that are not numbers: {want}")
        self.last = want
        return want.copy()
