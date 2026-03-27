"""Commands. Run `python -m so101.cli <command> --help` for the options of each."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Config, sim


def cmd_prepare(a):
    from .lerobot_io import prepare
    n = prepare(a.dataset, a.out, sim().img if a.sim else Config().img)
    print(f"{n} episodes written to {a.out}")


def cmd_train(a):
    from . import train as T
    from .data import Examples, Store
    cfg = sim() if a.sim else Config()
    try:
        if a.steps is not None:
            cfg = replace(cfg, steps=a.steps)
        if a.lr is not None:
            cfg = replace(cfg, lr=a.lr)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    store = Store(a.data)
    if store.img != cfg.img:
        raise SystemExit(f"{a.data} holds {store.img}-pixel pictures but training is set for {cfg.img}; "
                         f"run prepare and train both with --sim or both without")
    ex = Examples(cfg, store)
    print(f"{len(store.tasks)} episodes, {len(store.state)} frames at {store.fps} fps, cameras {store.cameras}")
    print("commands:", "; ".join(f"{c} ({store.tasks.count(c)})" for c in store.commands))
    model = T.train(cfg, ex, init=a.init)
    report = T.held_out_check(cfg, model, ex)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    T.save(a.out, cfg, model, ex)
    Path(a.out).with_suffix(".check.json").write_text(json.dumps(report, indent=1))
    print(f"saved {a.out}")


def cmd_bench(a):
    """How long one call to the world action model takes on this machine."""
    from . import train as T
    from .run import Policy
    cfg, model, info = T.load(a.model, T.pick_device(Config()))
    images = {c: np.zeros((480, 640, 3), dtype=np.uint8) for c in info["cameras"]}
    state = (info["norm"].low + info["norm"].high) / 2
    for future in (False, True):
        policy = Policy(cfg, model, info, with_future=future)
        times = []
        for _ in range(12):
            policy.infer(images, state, 0.0, info["commands"][0])
            times.append(policy.last_seconds)
        print(f"future prediction {'on' if future else 'off'}: {1000 * np.median(times[2:]):.0f} ms per call "
              f"on {policy.device.type} ({cfg.chunk / info['fps']:.1f} s of movement per call)")


def cmd_sim_record(a):
    from .lerobot_io import LeRobotSink
    from .sim import SimArm, record
    cfg = sim()
    sink = LeRobotSink(a.out, SimArm.cameras, SimArm.motors, SimArm.fps, cfg.img)
    kept, thrown = record(sink, a.tasks, img=cfg.img, tail=cfg.tail, seed0=a.seed, p_messy=a.messy)
    sink.close()
    print(f"{kept} episodes written to {a.out} ({thrown} failed attempts thrown away)")


def cmd_sim_eval(a):
    from .evaluate import evaluate, table
    results = evaluate(a.model, a.episodes, latency=a.latency)
    print(table(results))
    Path(a.model).with_suffix(".eval.json").write_text(json.dumps(results, indent=1))


def cmd_run(a):
    from . import train as T
    from .arm import RealArm
    from .planner import ClaudePlanner, ListPlanner, ManualPlanner
    from .run import Policy, Stack
    # The model's own shape comes from the saved file. Everything that applies while the arm runs
    # comes from config.py as it is now, then from the options below.
    changes = {k: v for k, v in dict(max_speed=a.max_speed, ticks_per_frame=a.ticks_per_frame,
                                     planner_model=a.planner_model).items() if v is not None}
    try:
        cfg, model, info = T.load(a.model, T.pick_device(Config()), settings=replace(Config(), **changes))
    except ValueError as e:                                 # settings refused by Config
        raise SystemExit(str(e)) from e
    cams = parse_cameras(a.cameras)
    missing = set(info["cameras"]) - set(cams)
    if missing:
        raise SystemExit(f"the model was trained with cameras {info['cameras']}; --cameras is missing {sorted(missing)}")
    steps = [s.strip() for s in a.steps.split(";") if s.strip()]
    unknown = [s for s in steps if s not in info["commands"]]
    if a.planner == "list" and (unknown or not steps):
        raise SystemExit(f"--steps must be commands this model was trained on ({'; '.join(info['commands'])}); "
                         f"not one of them: {unknown or 'nothing given'}")
    arm = RealArm(a.port, {k: camera_source(v) for k, v in cams.items()}, info["fps"], cfg.ticks_per_frame,
                  robot_id=a.robot_id, use_degrees=not a.no_degrees, width=a.camera_width, height=a.camera_height,
                  dry_run=a.dry_run)
    planner = {"claude": lambda: ClaudePlanner(cfg), "manual": ManualPlanner,
               "list": lambda: ListPlanner(steps)}[a.planner]()
    print("commands this model knows:", "; ".join(info["commands"]))
    print(f"speed limit {cfg.max_speed:g} units a second, {info['fps'] * cfg.ticks_per_frame} controller ticks a second, "
          f"command timeout {cfg.command_timeout:g} s" + (f", planner {cfg.planner_model}" if a.planner == "claude" else ""))
    if a.dry_run:
        print("dry run: nothing is sent to the motors")
    history, stack = [], None
    arm.connect()
    try:
        stack = Stack(cfg, Policy(cfg, model, info, with_future=a.future), arm, log=print)
        history = stack.run(a.instruction, planner)
    except KeyboardInterrupt:
        print("\nstopped; the arm holds its last position")
        history = getattr(stack, "history", []) + [("stop", "interrupted at the keyboard")]
    except Exception as e:
        print(f"\nstopped by an error; the arm holds its last position: {e!r}")
        history = getattr(stack, "history", []) + [("stop", f"error: {e!r}")]
        raise
    finally:
        if stack is not None:
            try:
                stack.ease()                                # do not stall a motor while waiting for Enter
            except Exception as e:
                print(f"could not ease the arm: {e!r}")
        try:
            input("Support the arm, then press Enter to release the motors. ")
        except (KeyboardInterrupt, EOFError):
            print("\nreleasing the motors now")
        finally:
            # `planner_saw[i]` is what the planner said it saw before choosing `history[i]`.
            report = dict(instruction=a.instruction, planner=a.planner, history=history,
                          planner_saw=getattr(stack, "seen", []), stats=stack.stats if stack else {})
            print(json.dumps(report, indent=1), flush=True)
            try:
                if a.save_history:
                    save_history(a.save_history, report)
            finally:
                arm.close()


def save_history(path, report):
    """Add one run to `path` as one line of JSON, so a session of trials ends up in one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **report)) + "\n")


def parse_cameras(text):
    """'front=0, wrist=/dev/video2' -> {"front": "0", "wrist": "/dev/video2"}. Spaces around names are ignored."""
    cams = {}
    for pair in filter(str.strip, text.split(",")):
        name, sep, source = pair.partition("=")
        if not sep or not name.strip() or not source.strip():
            raise SystemExit(f'--cameras: "{pair.strip()}" is not name=device, for example "front=0"')
        cams[name.strip()] = source.strip()
    return cams


def camera_source(value):
    """A camera given on the command line: a device index such as 0, or a path such as /dev/video0."""
    value = value.strip()
    return int(value) if value.isdigit() else value


def main(argv=None):
    p = argparse.ArgumentParser(prog="so101")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="copy a LeRobot dataset into the form training reads")
    s.add_argument("--dataset", required=True, nargs="+", help="folders of LeRobot datasets, or their Hugging Face names")
    s.add_argument("--out", required=True)
    s.add_argument("--sim", action="store_true", help="use the simulated arm's settings")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("train", help="train the world action model")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="runs/so101/model.pt")
    s.add_argument("--steps", type=int)
    s.add_argument("--sim", action="store_true")
    s.add_argument("--init", help="a saved model to continue training from")
    s.add_argument("--lr", type=float)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("bench", help="time the world action model on this machine")
    s.add_argument("--model", default="runs/so101/model.pt")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("run", help="run the three layers on the real arm")
    s.add_argument("--model", default="runs/so101/model.pt")
    s.add_argument("--port", required=True, help="serial port of the follower arm")
    s.add_argument("--cameras", required=True, help='camera name=device index or path, for example "front=0,wrist=/dev/video2"')
    s.add_argument("--instruction", required=True)
    s.add_argument("--planner", choices=["claude", "manual", "list"], default="claude")
    s.add_argument("--steps", default="", help='for --planner list: commands separated by ";"')
    s.add_argument("--robot-id", default="follower", help="the id the arm was calibrated under")
    s.add_argument("--no-degrees", action="store_true", help="the dataset was recorded with use_degrees off")
    s.add_argument("--future", action="store_true", help="also generate the predicted future on each call")
    s.add_argument("--dry-run", action="store_true", help="read the arm and run the model, but send nothing")
    s.add_argument("--max-speed", type=float, help="fastest any joint may be commanded to move, units per second")
    s.add_argument("--ticks-per-frame", type=int, help="controller ticks per dataset frame; 1 if the loop falls behind")
    s.add_argument("--planner-model", help="the model the API planner uses")
    s.add_argument("--save-history", metavar="FILE", help="also add this run's commands, events and what the planner "
                   "said it saw to FILE, one line of JSON per run")
    s.add_argument("--camera-width", type=int, default=640, help="camera picture width; use what the demonstrations were recorded at")
    s.add_argument("--camera-height", type=int, default=480, help="camera picture height; use what the demonstrations were recorded at")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("sim-record", help="record demonstrations on the simulated stand-in arm")
    s.add_argument("--out", required=True)
    s.add_argument("--tasks", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--messy", type=float, default=0.35, help="share of episodes that start from an awkward place")
    s.set_defaults(fn=cmd_sim_record)

    s = sub.add_parser("sim-eval", help="score a model on the simulated stand-in arm")
    s.add_argument("--model", default="runs/so101_sim/model.pt")
    s.add_argument("--episodes", type=int, default=50)
    s.add_argument("--latency", type=int, default=3, help="controller ticks the model's answer arrives late by")
    s.set_defaults(fn=cmd_sim_eval)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
