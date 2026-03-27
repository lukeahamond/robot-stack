"""Scoring on the simulated stand-in arm: two planners, with and without a forced drop."""
from . import train as T
from .planner import ListPlanner
from .run import Delayed, Policy, Stack
from .sim import SeeingPlanner, SimArm, fixed_list

ROWS = [("sees the scene (simulator truth)", True, False), ("follows a fixed list, never looks", False, False),
        ("sees the scene (simulator truth)", True, True), ("follows a fixed list, never looks", False, True)]


def evaluate(model_path, n=50, latency=3, seed0=5_000_000, with_future=False, log=print):
    cfg, model, info = T.load(model_path, "cpu")
    arm = SimArm(cfg.img)
    try:
        return _evaluate(cfg, arm, Policy(cfg, model, info, with_future=with_future), n, latency, seed0, log)
    finally:
        arm.close()


def _evaluate(cfg, arm, policy, n, latency, seed0, log):
    out = []
    for name, sees, drop in ROWS:
        tally = dict(success=0, commands=0, model_calls=0, surprises=0, seconds=0.0, dropped=0)
        for e in range(n):
            arm.reset(seed0 + e, drop=drop)
            stack = Stack(cfg, policy, arm, runner=Delayed(latency))
            planner = SeeingPlanner(arm) if sees else ListPlanner(fixed_list(arm))
            history = stack.run(arm.instruction, planner)
            tally["success"] += bool(arm.success)
            tally["dropped"] += bool(arm.dropped)
            tally["commands"] += sum(1 for c, _ in history if c not in ("done", "stop"))
            tally["model_calls"] += stack.stats["model_calls"]
            tally["surprises"] += stack.stats["surprises"]
            tally["seconds"] += stack.stats["ticks"] / stack.tick_hz
        row = dict(planner=name, disturbance="one forced drop" if drop else "none", episodes=n,
                   **{k: v / n for k, v in tally.items()})
        log(f"{name}, {row['disturbance']}: success {row['success']:.2f}")
        out.append(row)
    return out


def table(rows):
    lines = ["| Planner | Disturbance | Success | Commands | Model calls | Surprises | Seconds |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['planner'].capitalize()} | {r['disturbance']} | {r['success']:.2f} | {r['commands']:.1f} | "
                     f"{r['model_calls']:.1f} | {r['surprises']:.2f} | {r['seconds']:.1f} |")
    return "\n".join(lines)
