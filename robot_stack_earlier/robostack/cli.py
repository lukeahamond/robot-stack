"""Command line entry point.

    python3 -m robot_stack_earlier.robostack.cli all            collect, train, evaluate, RL fine-tune, evaluate again
    python3 -m robot_stack_earlier.robostack.cli all --no-rl    stop after the first evaluation
    python3 -m robot_stack_earlier.robostack.cli all --quick    tiny settings, to check the plumbing (about a minute and a half)
    python3 -m robot_stack_earlier.robostack.cli eval           re-evaluate a saved run
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config, quick
from .data import collect
from .models import Models
from .runtime import evaluate, format_table
from .train import rl_finetune, train_all


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robot_stack_earlier.robostack.cli")
    ap.add_argument("command", choices=["all", "eval"])
    ap.add_argument("--run", default="runs/default", help="directory for models and results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-rl", action="store_true")
    ap.add_argument("--episodes", type=int, default=500, help="evaluation episodes per variant")
    ap.add_argument("--device", default="auto",
                    help='"auto" trains the encoder on a GPU if there is one and runs the rest on the CPU; '
                         'any other value (for example "cpu") puts everything there')
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    if args.command == "eval" and (args.quick or args.seed is not None or args.no_rl):
        ap.error("eval uses the settings saved with the run; --quick, --seed and --no-rl only apply to all")
    if args.seed is not None and args.seed < 0:
        ap.error("--seed must be 0 or more")
    if args.episodes < 1:
        ap.error("--episodes must be 1 or more")

    device = "cpu" if args.device == "auto" else args.device
    cfg = Config(device=device, encoder_device=args.device, seed=args.seed or 0)
    if args.quick:
        cfg = quick(cfg)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    results = {}

    def report(tag, models):
        t0 = time.time()
        results[tag] = evaluate(cfg, models, n=args.episodes)
        print(f"\n== {tag} ({args.episodes} episodes per variant, {time.time() - t0:.0f}s) ==")
        print(format_table(results[tag]))
        (run / "results.json").write_text(json.dumps(results, indent=2))

    if args.command == "eval":
        saved = torch.load(run / "models.pt", map_location=device)
        cfg = Config(**{**saved["config"], "device": device, "encoder_device": args.device})
        models = Models(cfg).to(cfg.device)
        models.load_state_dict(saved["state"])
        report("saved models", models)
        return results

    t0 = time.time()
    eps = collect(cfg)
    models, buf = train_all(cfg, eps)
    print(f"trained in {time.time() - t0:.0f}s")
    report("after supervised training", models)
    if not args.no_rl:
        rl_finetune(cfg, models, buf)
        report("after RL fine-tuning", models)
    torch.save({"config": asdict(cfg), "state": models.state_dict()}, run / "models.pt")
    print(f"\nsaved to {run}/")
    return results


if __name__ == "__main__":
    main()
