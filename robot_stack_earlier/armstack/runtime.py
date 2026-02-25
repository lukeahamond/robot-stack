"""The three layers running together, and the harness that measures them.

    planner            called only on events; answers with a command in words
    world action model called every `exec_h` actions (0.8 s); generates frames, actions, expected touch
    controller         50 Hz, inside the simulator step

The lower two layers run a command until one of these events hands control back to the
planner: the model says the command is finished; the hand was holding something, the model
expected to keep feeling it, and the touch went away ("surprise"); or the command has run
too long. A grasp that simply misses is not a surprise: the model retries by itself.
"""
import numpy as np
import torch

from .control import LearnedController, clip_norm
from .data import norm_prop
from .sim import COLORS, DONE, HELD_TOUCH, NONE, PICK, PLACE, ArmEnv


class Stack:
    def __init__(self, cfg, ctrl_state, model, seed=0):
        self.cfg, self.model = cfg, model
        self.env = ArmEnv(cfg, LearnedController(cfg, ctrl_state))
        self.gen = torch.Generator()
        self.reset(seed)

    def reset(self, seed, drop=False):
        self.env.reset(seed, drop)
        self.gen.manual_seed(seed)
        self.stats = dict(actions=0, model_calls=0, planner_calls=0, surprises=0)
        self.trace = []          # per model call: (action index, frame seen, frames generated)
        self.seen = {}           # action index -> camera frame, kept only when recording
        return self

    def run_command(self, sg, record=False):
        env, c = self.env, self.cfg
        used = missing = 0
        holding = env.touch > HELD_TOUCH
        while True:
            if env.wrong:
                return "wrong block on the tray"
            if env.t >= c.max_actions:
                return "out of time"
            if used >= c.subgoal_timeout:
                return "timeout"
            seen = env.frame()
            frame = torch.from_numpy(seen).float()[None] / 127.5 - 1
            prop = torch.from_numpy(norm_prop(env.proprio()))[None]
            frames, act = self.model.sample(frame, prop, torch.tensor([sg]), generator=self.gen)
            act = act[0].numpy().astype(float)
            self.stats["model_calls"] += 1
            if record:
                self.trace.append((env.t, seen, ((frames[0].numpy() + 1) * 127.5).round().astype(np.uint8)))
            start = env.ee
            for k in range(c.exec_h):
                if act[k, 5] > 0:
                    return "finished"
                target = start + c.pos_scale * act[k, :3]
                target = env.ee + clip_norm(target - env.ee, 2 * c.max_step)       # no wild jumps
                target = np.clip(target, [0.15, -0.38, 0.008], [0.58, 0.38, 0.25])
                env.step(target, act[k, 3] > 0)
                if record:
                    self.seen[env.t] = env.frame()
                used += 1
                self.stats["actions"] += 1
                holding, missing = count_missing(holding, missing, env.touch, act[k, 4] > 0)
                if missing >= c.surprise_steps:
                    self.stats["surprises"] += 1
                    return "surprise"
                if env.wrong or env.t >= c.max_actions:
                    break


# ---- planners that need no language model (for measuring the lower two layers) ----
class SeeingPlanner:
    """Reads the simulator's true state. An upper bound on what a planner can do."""

    def __init__(self, env):
        pass

    def __call__(self, env, event):
        sg = env.oracle_subgoal()
        return None if sg[0] == DONE else sg


class BlindPlanner:
    """Turns the instruction into a fixed list and never looks at the scene. It moves on
    when a command reports finished and repeats the command on any other event."""

    def __init__(self, env):
        self.todo = [sg for i in env.targets for sg in ((PICK, int(env.color[i])), (PLACE, NONE))]
        self.i = 0

    def __call__(self, env, event):
        if event == "finished":
            self.i += 1
        return self.todo[self.i] if self.i < len(self.todo) else None


PLANNERS = {"planner that sees the scene": SeeingPlanner, "planner that follows a fixed list": BlindPlanner}


def count_missing(holding, missing, touch, expects_touch):
    """-> (holding, missing). `missing` counts consecutive actions of "was holding, expected touch,
    felt none"; any other action starts the count again. Letting go when no touch is expected ends
    the holding, so touch expected again later is not counted as a drop. "Felt" uses the same threshold as the
    demonstrations' "holding", which is what the expected-touch output was trained on."""
    if touch > HELD_TOUCH:
        return True, 0
    if holding and expects_touch:
        return holding, missing + 1
    if holding:
        return False, 0          # a planned release: touch is gone and nothing expects it
    return holding, 0


def run_episode(stack, make_planner, seed, drop=False):
    stack.reset(seed, drop)
    env, planner, event = stack.env, make_planner(stack.env), "start"
    while stack.stats["planner_calls"] < stack.cfg.max_planner_calls:
        sg = planner(env, event)
        stack.stats["planner_calls"] += 1
        if sg is None:
            break
        event = stack.run_command(sg)
        if event in ("wrong block on the tray", "out of time"):
            break
    return dict(success=float(env.success), wrong=float(env.wrong), unfinished=float(not env.success and not env.wrong),
                dropped=float(env.dropped), **{k: float(v) for k, v in stack.stats.items()})


def evaluate(cfg, ctrl_state, model, n=100, seed0=5_000_000, log=print):
    stack = Stack(cfg, ctrl_state, model)
    results = {}
    try:
        for drop in (False, True):
            for name, make in PLANNERS.items():
                rows = [run_episode(stack, make, seed0 + i, drop) for i in range(n)]
                key = f"{name}, {'one forced drop' if drop else 'no disturbance'}"
                results[key] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                results[key]["episodes"] = n
                log(f"{key}: success {results[key]['success']:.2f}")
    finally:
        stack.env.close()
    return results


def format_table(results):
    cols = [("success", "success"), ("wrong", "wrong block"), ("unfinished", "unfinished"), ("planner_calls", "planner calls"),
            ("model_calls", "model calls"), ("actions", "actions"), ("surprises", "surprises")]
    rows = [["variant"] + [h for _, h in cols]]
    for name, r in results.items():
        rows.append([name] + [f"{r[k]:.2f}" for k, _ in cols])
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([lines[0], "  ".join("-" * x for x in w)] + lines[1:])


def parse_command(text):
    """Words to the two tokens the world action model was trained on. None means stop."""
    t = text.lower()
    if any(w in t for w in ("done", "finished", "stop")):
        return None
    if any(w in t for w in ("pick", "grab", "take")):
        for i, name in enumerate(COLORS):
            if name in t:
                return (PICK, i)
        raise ValueError("a pick command has to name red, green or blue")
    if any(w in t for w in ("put", "place", "tray")):
        return (PLACE, NONE)
    raise ValueError(f"not a command the action model knows: {text!r}")
