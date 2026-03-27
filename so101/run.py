"""The three layers running together on an arm.

An "arm" is anything with these members, which both the real SO-101 and the simulated
stand-in provide:

    fps, cameras, motors
    read_state()  -> six joint values
    read_images() -> {camera name: (H, W, 3) uint8}
    send(target)     six joint targets; returns False if nothing was sent (a dry run)
    wait_tick()      let one controller tick of time pass

An arm may also count `late_ticks`, the ticks that started behind schedule, and report
`temperatures()`, each motor's temperature in degrees C by motor name.

Timing: the controller ticks `ticks_per_frame` times per dataset frame. The world action
model is slower than a tick, so its answer arrives some ticks after the observation it was
computed from. The plan is anchored at the observation's tick, the part that is already in
the past is skipped, and the new plan is blended in over a few frames.
"""
import threading
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

from .controller import GRIP, Controller, Plan, TouchSensor
from .planner import DONE, STOP, PlannerUnavailable

MODEL_ERROR = "model error"      # the world action model raised; the arm holds still and the run ends
DRAIN_SECONDS = 10.0             # longest wait for a model call left over from the last command
SPEED_WINDOW = 0.1               # seconds the gripper speed is measured over, so one noisy reading does not count
TEMPERATURE_MISSES = 3           # failed temperature readings in a row before the run ends: the heat check is then off


class Policy:
    """The world action model with its scaling, taking raw readings and returning joint targets."""

    def __init__(self, cfg, model, info, with_future=False):
        self.cfg, self.model, self.info = cfg, model, info
        self.norm = info["norm"]
        self.device = next(model.parameters()).device
        self.with_future = with_future
        self.amp = cfg.amp and self.device.type == "cuda"
        self.last_seconds = 0.0

    @torch.no_grad()
    def infer(self, images, state, touch, command):
        """-> joint targets (chunk, 6), expected touch (chunk,) in -1..1, finished (chunk,) in -1..1."""
        t0 = time.perf_counter()
        size = self.cfg.img
        frames = [np.asarray(Image.fromarray(images[c]).resize((size, size), Image.BILINEAR)) if images[c].shape[:2] != (size, size)
                  else images[c] for c in self.model.cameras]
        x = torch.from_numpy(np.stack(frames)).to(self.device).permute(0, 3, 1, 2).float()[None] / 255.0
        prop = torch.from_numpy(self.norm.prop(np.asarray(state, dtype=np.float32), touch))[None].to(self.device)
        cmd = self.model.command_ids([command]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            act, _ = self.model.sample(x, prop, cmd, with_future=self.with_future)
        chunk = act[0].float().cpu().numpy()
        self.last_seconds = time.perf_counter() - t0
        return self.norm.targets(chunk, np.asarray(state, dtype=np.float64)), chunk[:, 6], chunk[:, 7]


class Delayed:
    """Runs the model at once and releases the answer `ticks` later. For simulation."""

    def __init__(self, ticks):
        self.ticks = ticks

    def submit(self, fn, tick):
        self.value = self.error = None
        self.at = tick + self.ticks
        try:
            self.value = fn()
        except Exception as e:           # raised by ready(), as Threaded does
            self.error = e

    def ready(self, tick):
        if tick < self.at:
            return False
        if self.error is not None:
            raise self.error
        return True


class Threaded:
    """Runs the model on another thread so the controller keeps ticking. For the real arm.

    Each call gets its own job, so a call that is still running when the next is submitted
    can never hand its answer to the new one.
    """

    def submit(self, fn, tick):
        self.job = job = dict(value=None, error=None)
        job["thread"] = threading.Thread(target=self._run, args=(fn, job), daemon=True)
        job["thread"].start()

    @staticmethod
    def _run(fn, job):
        try:
            job["value"] = fn()
        except Exception as e:           # surfaced on the control thread by ready()
            job["error"] = e

    @property
    def value(self):
        return self.job["value"]

    def ready(self, tick):
        if self.job["thread"].is_alive():
            return False
        if self.job["error"] is not None:
            raise self.job["error"]
        return True


class Stack:
    def __init__(self, cfg, policy, arm, runner=None, log=None):
        self.cfg, self.policy, self.arm = cfg, policy, arm
        self.runner = runner or Threaded()
        self.log = log or (lambda *_: None)
        nm = policy.norm
        self.controller = Controller(cfg, arm.fps, nm.low, nm.high)
        self.sensor = TouchSensor(cfg)
        self.tick_hz = arm.fps * cfg.ticks_per_frame
        self.tick = 0
        self.holding = False          # kept between commands: a block picked up by one is carried by the next
        self.unread = 0               # temperature readings that have failed in a row
        self.stats = dict(model_calls=0, ticks=0, surprises=0, late_ticks=0, model_errors=0)

    def run_command(self, command, on_tick=None):
        """Run one command until it ends. Returns "finished", "surprise", "timeout" or "model error"."""
        c, arm, ctl = self.cfg, self.arm, self.controller
        # An arm with a clock to restart runs in real time, so the timeout is also measured on the wall clock:
        # counting ticks alone lets a loop that runs behind go on far longer than `command_timeout`. The
        # simulated arm has no clock and runs as fast or slow as the computer allows, so it counts ticks only.
        realtime = hasattr(arm, "restart_clock")
        if realtime:
            arm.restart_clock()
        began = time.monotonic()
        state = arm.read_state()
        ctl.reset(state)
        pending = False
        holding_ticks = missing_ticks = votes = 0
        grips = deque(maxlen=max(2, round(SPEED_WINDOW * self.tick_hz) + 1))   # recent gripper readings
        plan = None
        start = self.tick
        late0 = getattr(arm, "late_ticks", 0)
        outcome = None
        while outcome is None:
            state = arm.read_state()
            touch = self.sensor.read(state[GRIP])
            # ---- ask for a new chunk when the current one has run far enough ----
            if not pending and (plan is None or plan.frame(self.tick) >= c.execute):
                images = arm.read_images()
                self.runner.submit(lambda s=state.copy(), t=touch, im=images: self.policy.infer(im, s, t, command), self.tick)  # noqa: B008 (the defaults freeze this tick's readings)
                pending, asked_at = True, self.tick
            try:
                arrived = pending and self.runner.ready(self.tick)
                if arrived and not all(np.isfinite(np.asarray(v, dtype=float)).all() for v in self.runner.value):
                    raise ValueError("its answer held values that are not numbers (NaN or infinity)")
            except Exception as e:                            # the model call failed: hold still and hand back
                self.log(f"  the world action model failed: {e}")
                pending, arrived, outcome = False, False, MODEL_ERROR
                self.stats["model_errors"] += 1
                break                                         # send nothing more; the servos hold the last target
            if arrived:
                targets, exp_touch, finished = self.runner.value
                plan = Plan(asked_at, targets, exp_touch, finished, c.ticks_per_frame)
                ctl.set_plan(plan, self.tick)
                pending = False
                self.stats["model_calls"] += 1
                # Vote on the frames still to come: the ones before the answer arrived are already past.
                now = min(int(plan.frame(self.tick)), len(finished) - 1)
                votes = votes + 1 if finished[now:now + c.execute].mean() > 0 else 0
                if votes >= c.finish_votes:
                    outcome = "finished"
            # ---- act ----
            target = ctl.tick(self.tick, state)
            if arm.send(target) is not False:                 # a dry run sends nothing, so nothing counts as sent
                self.sensor.sent_command(target[GRIP])
            # ---- the surprise alarm: was holding, the plan expects touch, the hand feels none ----
            grips.append(state[GRIP])
            grip_speed = abs(grips[-1] - grips[0]) * self.tick_hz / (len(grips) - 1) if len(grips) > 1 else 0.0
            holding_ticks = holding_ticks + 1 if touch > 0.5 and grip_speed < c.hold_still else 0
            if holding_ticks >= c.hold_time * self.tick_hz:
                self.holding = True
            # Before the first chunk of a command there is no plan, and nothing has asked for a release.
            # While a new chunk blends in, the expectation follows whichever chunk the targets mostly come from.
            expected = ctl.expected_touch(self.tick)
            expects = expected > 0 if expected is not None else self.holding
            if self.holding and not expects and touch < 0.2:
                self.holding = False                          # a planned release
            missing_ticks = missing_ticks + 1 if (self.holding and expects and touch < 0.2) else 0
            if missing_ticks >= c.surprise_time * self.tick_hz:
                outcome = "surprise"
                self.holding = False
                self.stats["surprises"] += 1
            if outcome is None and ((self.tick - start) / self.tick_hz > c.command_timeout
                                    or (realtime and time.monotonic() - began > c.command_timeout)):
                outcome = "timeout"
            if on_tick:
                on_tick(self, target, touch)
            arm.wait_tick()
            self.tick += 1
            self.stats["ticks"] += 1
        ctl.hold()
        ran, late = self.tick - start, getattr(arm, "late_ticks", 0) - late0
        self.stats["late_ticks"] += late
        if late > 0.05 * ran:
            self.log(f"  the control loop ran behind on {late} of {ran} ticks, so movement played back slower than "
                     f"recorded; pass --ticks-per-frame 1")
        if pending:                                           # let a stray model call finish before the next command
            give_up = time.monotonic() + DRAIN_SECONDS
            try:
                while not self.runner.ready(self.tick + 10 ** 9):
                    if time.monotonic() > give_up:          # the model is stuck: end the run, holding still
                        self.log(f"  a model call left over from this command is still running after "
                                 f"{DRAIN_SECONDS:g} s; the world action model is stuck")
                        outcome = MODEL_ERROR
                        self.stats["model_errors"] += 1
                        break
                    time.sleep(0.005)
            except Exception as e:                            # its answer would not have been used
                self.log(f"  a model call left over from this command failed and was dropped: {e}")
        return outcome

    def ease(self):
        """Push no harder than the idle limits while nothing is running.

        Between commands no ticks run and the servos hold the last target. That target may squeeze the
        gripper up to `grip_squeeze` past where it is, and may aim a joint that is held against something
        up to `max_lead` past where it is. Held for a whole planner call, either stalls a motor and heats
        it. The gripper is eased to `idle_squeeze`, which still holds the block and still reads as full
        touch, and each joint to `idle_lead`. The targets only move towards where the arm already is.
        """
        ctl, c = self.controller, self.cfg
        if ctl.last is None:
            return
        state = np.asarray(self.arm.read_state(), dtype=float)
        target = ctl.last.copy()
        target[:GRIP] = state[:GRIP] + np.clip(target[:GRIP] - state[:GRIP], -c.idle_lead, c.idle_lead)
        target[GRIP] = max(target[GRIP], state[GRIP] - c.idle_squeeze)
        if not np.isfinite(target).all():                     # a bad reading: keep the last good target
            self.log(f"  could not ease the arm: it read {state}")
            return
        if np.array_equal(target, ctl.last):
            return
        if self.arm.send(target) is not False:
            self.sensor.sent_command(target[GRIP])
        ctl.last = target

    def too_hot(self):
        """A reason to stop if the arm reports a motor at or above its limit (`grip_hot` for the gripper,
        `joint_hot` for the joints), else None.

        One failed reading is reported and the run goes on. `TEMPERATURE_MISSES` in a row is a reason to
        stop: on a bus that has stopped answering, the heat check would otherwise be off without a word.
        """
        read = getattr(self.arm, "temperatures", None)
        if read is None:
            return None
        try:
            celsius = {str(m): float(v) for m, v in read().items()}
        except Exception as e:
            self.unread += 1
            self.log(f"  could not read the motors' temperatures: {e}")
            if self.unread >= TEMPERATURE_MISSES:
                return f"the motors' temperatures could not be read {self.unread} times in a row"
            return None
        self.unread = 0
        for motor, value in celsius.items():
            name = "grip_hot" if motor == "gripper" else "joint_hot"
            if value >= getattr(self.cfg, name):
                return f"{motor} motor at {value:g} C ({name} is {getattr(self.cfg, name):g} C)"
        return None

    def run(self, instruction, planner, commands=None, on_tick=None):
        """Planner in charge: ask for a command, run it, report back, repeat. Returns the history."""
        commands = commands or self.policy.info["commands"]
        self.history = history = []                           # kept on the stack so an interrupted run can report it
        self.seen = seen = []                                 # what the planner said it saw, before each choice
        while len(history) < self.cfg.max_commands:
            self.ease()                                       # the planner may take a while
            try:
                command = planner.next(instruction, commands, history, self.arm.read_images())
            except PlannerUnavailable as e:
                self.log(f"planner unavailable ({e}); holding still")
                history.append((STOP, f"planner unavailable: {e}"))
                seen.append("")
                break
            seen.append(getattr(planner, "last_observation", ""))
            if seen[-1]:
                self.log(f"planner sees: {seen[-1]}")
            if command in (DONE, STOP):
                history.append((command, "ended by the planner"))
                break
            # The arm held its last target for the whole planner call, which can be most of a minute with a
            # block squeezed, so the motors are checked again before anything moves.
            hot = self.too_hot()
            if hot:
                self.log(f"{hot}; holding still")
                history.append((STOP, hot))
                break
            if command not in commands:                       # the model would drop the words it does not know
                self.log(f"\"{command}\" is not a command this model was trained on; holding still")
                history.append((STOP, f"not a command the arm has: {command}"))
                break
            self.log(f"command: {command}")
            outcome = self.run_command(command, on_tick)
            self.log(f"  -> {outcome}")
            history.append((command, outcome))
            if outcome == MODEL_ERROR:
                self.log("the world action model failed; holding still")
                break
            hot = self.too_hot()
            if hot:
                self.log(f"{hot}; holding still")
                history.append((STOP, hot))
                break
        else:
            self.log(f"stopped after {self.cfg.max_commands} commands (max_commands) without the planner saying done")
            history.append((STOP, f"stopped after {self.cfg.max_commands} commands (max_commands)"))
            seen.append("")
        self.ease()
        return history
