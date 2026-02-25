"""Command line.

    python3 -m robot_stack_earlier.armstack.cli train            train the controller, record demonstrations, train the world action model
    python3 -m robot_stack_earlier.armstack.cli eval             run the trained stack with the scripted planners
    python3 -m robot_stack_earlier.armstack.cli train --quick    tiny settings, to check the plumbing
"""
import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .config import Config, quick
from .control import train_controller
from .data import Dataset, collect
from .train import held_out_report, train_wam


def log(*a):
    print(*a, flush=True)


def load(run, device="cpu"):
    from .wam import WAM
    saved = torch.load(Path(run) / "model.pt", map_location=device, weights_only=True)   # no pickled code
    cfg = Config(**saved["config"])
    model = WAM(cfg).to(device)
    model.load_state_dict(saved["wam"])
    return cfg, saved["controller"], model.eval()


# Settings that only change the world action model or how it is run, not the recorded demonstrations.
NOT_IN_DEMOS = {"exec_h", "patch", "fut_down", "width", "depth", "heads", "batch", "lr", "wam_steps", "w_action",
                "denoise_steps", "subgoal_timeout", "surprise_steps", "max_planner_calls", "device", "workers"}
DEMO_SOURCES = ("config.py", "sim.py", "expert.py", "control.py", "data.py")


def demo_seed0(cfg):
    """First simulator seed of the demonstrations. Each --seed records its own episodes, and none
    reach the evaluation seeds (from 5,000,000) for seed 0, or overlap them for any other seed."""
    return cfg.seed * 10_000_000


def demo_cache():
    """A folder only this user can write to. The cached demonstrations are a pickle, and loading a
    pickle runs code, so they must never come from a shared folder such as /tmp."""
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "armstack"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def demo_key(cfg):
    """A name for the cached demonstrations that changes with every setting and every line of code they depend on."""
    h = hashlib.sha256(json.dumps({k: v for k, v in asdict(cfg).items() if k not in NOT_IN_DEMOS},
                                  sort_keys=True, default=list).encode())
    h.update(__package__.encode())      # the cache is pickled objects of this package: another package name cannot load it
    for name in DEMO_SOURCES:
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robot_stack_earlier.armstack.cli")
    ap.add_argument("command", choices=["train", "eval"])
    ap.add_argument("--run", default="runs/arm")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=None, help="evaluation episodes per variant (default 100)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="training steps for the world action model")
    args = ap.parse_args(argv)
    if args.command == "eval" and (args.quick or args.seed is not None or args.steps is not None):
        ap.error("eval uses the settings saved with the run; --quick, --seed and --steps only apply to train")
    if args.command == "train" and args.episodes is not None:
        ap.error("--episodes only applies to eval")
    if args.episodes is not None and args.episodes < 1:
        ap.error("--episodes must be 1 or more")
    if args.seed is not None and args.seed < 0:
        ap.error("--seed must be 0 or more")
    if args.steps is not None and args.steps < 1:
        ap.error("--steps must be 1 or more")
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    if args.command == "train":
        cfg = Config(seed=args.seed or 0)
        if args.quick:
            cfg = quick(cfg)
        if args.steps is not None:
            cfg = replace(cfg, wam_steps=args.steps)
        t0 = time.time()
        ctrl = train_controller(cfg, log)
        cache = demo_cache() / f"demos_{demo_key(cfg)}.pt"
        if cache.exists():                    # demonstrations are slow to record, so reuse them
            ds = torch.load(cache, weights_only=False)
            log(f"demonstrations reused from {cache}")
        else:
            ds = Dataset(cfg, collect(cfg, ctrl, log, seed0=demo_seed0(cfg)))
            torch.save(ds, cache, pickle_protocol=4)
        log(f"training examples {len(ds.train_idx)}, held out {len(ds.held_idx)}")
        model = train_wam(cfg, ds, log)
        report = held_out_report(cfg, model, ds, log)
        torch.save({"config": asdict(cfg), "controller": ctrl, "wam": model.state_dict(), "held_out": report},
                   run / "model.pt")
        log(f"saved to {run}/ after {time.time() - t0:.0f}s")
        return

    from .runtime import evaluate, format_table
    cfg, ctrl, model = load(run)
    results = evaluate(cfg, ctrl, model, n=args.episodes or 100, log=log)
    log(format_table(results))
    (run / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
