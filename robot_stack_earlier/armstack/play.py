"""Drive the stack with a planner that is not in the code: someone looks at the picture
and types the next command. Each call replays the episode from the start (it is
deterministic) and runs the newest command until the next event.

    python3 -m robot_stack_earlier.armstack.play new --seed 7 --drop
    python3 -m robot_stack_earlier.armstack.play say "pick up the red block"
    python3 -m robot_stack_earlier.armstack.play say done
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .cli import load
from .runtime import Stack, parse_command


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robot_stack_earlier.armstack.play")
    ap.add_argument("command", choices=["new", "say"])
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--run", default="runs/arm")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drop", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "say" and (args.seed is not None or args.drop):
        ap.error("--seed and --drop choose the episode, so they go with `new`; `say` continues the episode already started")
    out = Path(args.run) / "play"
    out.mkdir(parents=True, exist_ok=True)
    session_file = out / "session.json"
    if args.command == "new":
        # Demonstrations used seeds below 1,500 and evaluation starts at 5,000,000, so the default is a scene not trained on.
        session = dict(seed=5_000_000 if args.seed is None else args.seed, drop=args.drop, commands=[], checks=[], closed=False)
    else:
        if not session_file.exists():
            raise SystemExit("no episode started yet; start one with `new`")
        session = json.loads(session_file.read_text())
        try:
            parse_command(args.text)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        if session["closed"]:
            raise SystemExit("this episode is over; start another with `new`")
        session["commands"].append(args.text)

    cfg, ctrl, model = load(args.run)
    stack = Stack(cfg, ctrl, model)
    stack.reset(session["seed"], session["drop"])
    env, event = stack.env, "start"
    for n, text in enumerate(session["commands"]):
        sg = parse_command(text)
        if sg is None:
            event = "stopped by the planner"
            break
        event = stack.run_command(sg)
        check = hashlib.md5(np.round(env.data.qpos, 6).tobytes()).hexdigest()[:8]
        if n < len(session["checks"]):
            if session["checks"][n] != check:
                raise SystemExit("replay did not reproduce the earlier state")
        else:
            session["checks"].append(check)

    Image.fromarray(env.frame(256)).save(out / "view.png")
    print(f"instruction: {env.instruction}")
    print(f"event: {event}   actions so far: {env.t} of {cfg.max_actions}   planner calls: {len(session['commands'])}")
    over = event in ("stopped by the planner", "wrong block on the tray", "out of time")
    if over:
        session["closed"] = True
        result = dict(seed=session["seed"], drop=session["drop"], dropped=bool(env.dropped), commands=session["commands"],
                      success=bool(env.success), wrong=bool(env.wrong), actions=env.t, surprises=stack.stats["surprises"])
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"episode over: success={env.success} wrong={env.wrong} forced drop happened={env.dropped}")
    session_file.write_text(json.dumps(session))


if __name__ == "__main__":
    main()
