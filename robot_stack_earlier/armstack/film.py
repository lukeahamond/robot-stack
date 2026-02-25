"""Save a picture comparing the frames the world action model generated with what the
camera showed 0.8 s later.

The model was trained to show the frame where the command ends if that comes sooner, so
only model calls with at least 0.8 s of their command still to run are shown.

    python3 -m robot_stack_earlier.armstack.film --seed 5000003 --drop
"""
import argparse
import bisect
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cli import load
from .runtime import SeeingPlanner, Stack


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robot_stack_earlier.armstack.film")
    ap.add_argument("--run", default="runs/arm")
    ap.add_argument("--seed", type=int, default=5_000_003)
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--columns", type=int, default=8)
    args = ap.parse_args(argv)
    cfg, ctrl, model = load(args.run)
    stack = Stack(cfg, ctrl, model).reset(args.seed, args.drop)
    env, planner, event = stack.env, SeeingPlanner(stack.env), "start"
    ends = []                    # action count at the end of each command
    for _ in range(cfg.max_planner_calls):
        sg = planner(env, event)
        if sg is None:
            break
        event = stack.run_command(sg, record=True)
        ends.append(env.t)
        if event in ("wrong block on the tray", "out of time"):
            break
    ahead = cfg.future[0]
    calls = [(t, seen, gen) for t, seen, gen in stack.trace
             if t + ahead in stack.seen and t + ahead <= ends[bisect.bisect_right(ends, t)]]
    if not calls:
        raise SystemExit("no model call had 0.8 s of its command left to run, so there is nothing to compare")
    pick = np.linspace(0, len(calls) - 1, min(args.columns, len(calls))).round().astype(int)
    size, pad, top = 144, 6, 22
    rows = ["camera now", f"generated: {ahead / 5:.1f} s ahead", f"camera {ahead / 5:.1f} s later"]
    sheet = Image.new("RGB", (len(pick) * (size + pad) + pad, len(rows) * (size + top) + pad), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for col, i in enumerate(pick):
        t, seen, gen = calls[i]
        for row, img in enumerate((seen, gen[0], stack.seen[t + ahead])):
            x, y = pad + col * (size + pad), row * (size + top) + top
            sheet.paste(Image.fromarray(img).resize((size, size), Image.BILINEAR), (x, y))
            if col == 0:
                draw.text((x, y - 16), rows[row], fill=(230, 230, 230))
    out = Path(args.run) / "generated_vs_actual.png"
    sheet.save(out)
    print(f"{env.instruction} success={env.success}; saved {out}")


if __name__ == "__main__":
    main()
