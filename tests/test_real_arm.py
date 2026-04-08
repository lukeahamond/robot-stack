"""Checks for so101/arm.py with a stand-in for LeRobot, so they need no arm and no LeRobot install.

They cover what this repository's code does with the robot object. They do not show that
LeRobot or the arm behave as the stand-in does.
"""
import json
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from so101.arm import MOTORS


class FakeRobot:
    def __init__(self, config):
        self.config, self.sent, self.connected = config, [], False
        self.bus = SimpleNamespace(sync_read=self.sync_read)
        self.cameras = {name: SimpleNamespace(read_latest=lambda: np.zeros((480, 640, 3), dtype=np.uint8))
                        for name in config.cameras}

    def sync_read(self, name, motors=None, normalize=True, num_retry=0):
        if name == "Present_Temperature":
            assert normalize is False                   # a raw register
            return {m: 41 for m in motors or MOTORS}
        return {m: float(i) for i, m in enumerate(MOTORS)}

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def send_action(self, action):
        self.sent.append(action)


@pytest.fixture
def RealArm(monkeypatch):
    def module(name, **members):
        m = ModuleType(name)
        vars(m).update(members)
        monkeypatch.setitem(sys.modules, name, m)

    config = lambda **kw: SimpleNamespace(**kw)
    module("lerobot")
    module("lerobot.cameras")
    module("lerobot.cameras.opencv", OpenCVCameraConfig=config)
    module("lerobot.robots")
    module("lerobot.robots.so_follower", SO101Follower=FakeRobot, SO101FollowerConfig=config)
    from so101.arm import RealArm
    return RealArm


def test_reads_and_sends_in_motor_order(RealArm):
    arm = RealArm("/dev/null", {"front": 0, "wrist": 2}, fps=30, ticks_per_frame=2)
    arm.connect()
    assert arm.robot.connected and arm.robot.config.use_degrees
    assert arm.read_state().tolist() == [0, 1, 2, 3, 4, 5]
    assert set(arm.read_images()) == {"front", "wrist"}
    assert arm.send(np.arange(6) * 10.0) is True
    assert arm.robot.sent == [{f"{m}.pos": 10.0 * i for i, m in enumerate(MOTORS)}]
    assert arm.temperatures() == {m: 41.0 for m in MOTORS}
    arm.close()
    assert not arm.robot.connected


def test_dry_run_sends_nothing_and_says_so(RealArm):
    arm = RealArm("/dev/null", {"front": 0}, fps=30, ticks_per_frame=2, dry_run=True)
    arm.connect()
    assert arm.send(np.zeros(6)) is False and arm.robot.sent == []


def test_ticks_that_start_behind_schedule_are_counted(RealArm):
    arm = RealArm("/dev/null", {"front": 0}, fps=5, ticks_per_frame=2)    # slow ticks: a busy machine is not "late"
    arm.connect()
    arm.wait_tick()
    assert arm.late_ticks == 0
    time.sleep(3 * arm.period)                         # the loop body took three ticks' worth of time
    arm.wait_tick()
    assert arm.late_ticks == 1


# ---- the `run` command, end to end, with the stand-in LeRobot -----------------------------

@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    from so101 import train as T
    from so101.config import quick
    from so101.data import Examples, Store, StoreWriter
    from so101.sim import SimArm, record
    root = tmp_path_factory.mktemp("run")
    cfg = quick()
    writer = StoreWriter(root / "store", SimArm.cameras, SimArm.motors, SimArm.fps, cfg.img)
    record(writer, 2, img=cfg.img, tail=cfg.tail, log=lambda *_: None)
    writer.close()
    ex = Examples(cfg, Store(root / "store"))
    T.save(root / "model.pt", cfg, T.train(cfg, ex, log=lambda *_: None, device="cpu"), ex)
    return root / "model.pt", ex.store.commands


def run_cli(monkeypatch, capsys, model, extra):
    import json
    from dataclasses import replace
    from so101 import cli
    from so101.config import Config
    monkeypatch.setattr(cli, "Config", lambda: replace(Config(), command_timeout=0.1, device="cpu"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    try:
        cli.main(["run", "--model", str(model), "--port", "/dev/null", "--cameras", "front=0,wrist=/dev/video2",
                  "--instruction", "tidy up"] + extra)
    finally:
        out = LAST_OUT[0] = capsys.readouterr().out
    return out, json.loads(out[out.index("{"):])


LAST_OUT = [""]


def test_the_run_command_drives_the_arm_and_reports(RealArm, saved_model, monkeypatch, capsys):
    model, commands = saved_model
    out, report = run_cli(monkeypatch, capsys, model, ["--planner", "list", "--steps", commands[0]])
    assert "commands this model knows" in out
    assert [c for c, _ in report["history"]][:1] == [commands[0]] and report["stats"]["ticks"] > 0


def test_the_run_command_reports_and_releases_after_an_error(RealArm, saved_model, monkeypatch, capsys):
    from so101 import run as R
    model, commands = saved_model

    def fail(*a, **k):
        raise OSError("serial port lost")
    monkeypatch.setattr(R.Stack, "run_command", fail)
    closed = []
    monkeypatch.setattr(FakeRobot, "disconnect", lambda self: closed.append(True))
    with pytest.raises(OSError):
        run_cli(monkeypatch, capsys, model, ["--planner", "list", "--steps", commands[0]])
    assert closed == [True]
    out = capsys.readouterr().out + LAST_OUT[0]
    report = json.loads(out[out.index("{"):])
    assert report["history"][-1][0] == "stop" and "serial port lost" in report["history"][-1][1]


def test_the_motors_are_released_if_the_stack_cannot_be_built(RealArm, saved_model, monkeypatch, capsys):
    from so101 import run as R
    model, commands = saved_model

    def fail(*a, **k):
        raise RuntimeError("no GPU")
    monkeypatch.setattr(R.Policy, "__init__", fail)
    closed = []
    monkeypatch.setattr(FakeRobot, "disconnect", lambda self: closed.append(True))
    with pytest.raises(RuntimeError):
        run_cli(monkeypatch, capsys, model, ["--planner", "list", "--steps", commands[0]])
    assert closed == [True]


def test_the_run_command_refuses_steps_it_was_not_trained_on(RealArm, saved_model, monkeypatch, capsys):
    model, _ = saved_model
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, capsys, model, ["--planner", "list", "--steps", "juggle"])


# ---- reading LeRobot datasets, with a stand-in -------------------------------------------

def fake_lerobot(monkeypatch, lengths, tasks, offset=0):
    """A stand-in LeRobot dataset. `offset` makes the episode table say the frames start later than they do."""
    import torch
    n = sum(lengths)
    starts = np.cumsum([0] + lengths) + offset

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, repo_id, root=None, return_uint8=False):
            self.meta = SimpleNamespace(
                camera_keys=["observation.images.front"], fps=30,
                features={"action": {"names": [f"{m}.pos" for m in MOTORS]}},
                episodes=[{"dataset_from_index": int(a), "dataset_to_index": int(b), "tasks": [t]}
                          for a, b, t in zip(starts[:-1], starts[1:], tasks, strict=True)])

        def __len__(self):
            return n

        def __getitem__(self, i):
            return {"observation.state": torch.full((6,), float(i)), "action": torch.full((6,), float(i) + 0.5),
                    "observation.images.front": torch.full((3, 30, 40), i % 256, dtype=torch.uint8),
                    "index": torch.tensor(i)}
    lerobot = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    datasets.LeRobotDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)


def test_prepare_copies_episodes_across_batch_boundaries(monkeypatch, tmp_path):
    from so101.data import Store
    from so101.lerobot_io import prepare
    lengths, tasks = [5, 23, 1, 9], ["pick up the red block", "put it in the box", "pick up the red block", "put it in the box"]
    n = sum(lengths)
    fake_lerobot(monkeypatch, lengths, tasks)
    (tmp_path / "ds").mkdir()
    assert prepare([str(tmp_path / "ds")], tmp_path / "store", 16, log=lambda *_: None) == 4
    s = Store(tmp_path / "store")
    assert s.tasks == tasks and s.lengths.tolist() == lengths and s.cameras == ["front"] and s.fps == 30
    assert s.state[:, 0].tolist() == list(range(n)) and (s.action[:, 0] == s.state[:, 0] + 0.5).all()
    assert s.frames["front"].shape == (n, 16, 16, 3) and s.frames["front"][17, 0, 0, 0] == 17


def test_restarting_the_clock_forgives_the_wait_before_a_command(RealArm):
    arm = RealArm("/dev/null", {"front": 0}, fps=5, ticks_per_frame=2)    # slow ticks: a busy machine is not "late"
    arm.connect()
    time.sleep(5 * arm.period)                         # the planner was thinking
    arm.restart_clock()
    arm.wait_tick()
    assert arm.late_ticks == 0


def test_prepare_refuses_frames_that_are_not_where_the_episode_table_says(monkeypatch, tmp_path):
    from so101.lerobot_io import prepare
    fake_lerobot(monkeypatch, [5, 7], ["pick up the red block", "put it in the box"], offset=2)
    (tmp_path / "ds").mkdir()
    with pytest.raises(ValueError, match="not contiguous"):
        prepare([str(tmp_path / "ds")], tmp_path / "store", 16, log=lambda *_: None)


def test_the_run_command_refuses_a_negative_speed_limit(RealArm, saved_model, monkeypatch, capsys):
    model, commands = saved_model
    with pytest.raises(SystemExit, match="max_speed must be above 0"):
        run_cli(monkeypatch, capsys, model, ["--planner", "list", "--steps", commands[0], "--max-speed", "-50"])


def test_the_run_command_saves_each_run_as_one_line(RealArm, saved_model, monkeypatch, capsys, tmp_path):
    model, commands = saved_model
    log = tmp_path / "session" / "runs.jsonl"
    for _ in range(2):
        _, report = run_cli(monkeypatch, capsys, model, ["--planner", "list", "--steps", commands[0],
                                                         "--save-history", str(log)])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 2 and lines[-1]["history"] == report["history"] and lines[-1]["instruction"] == "tidy up"
    assert len(lines[-1]["planner_saw"]) == len(report["history"]) and "time" in lines[-1]


def test_the_run_command_refuses_a_speed_limit_that_limits_nothing(RealArm, saved_model, monkeypatch, capsys):
    model, commands = saved_model
    with pytest.raises(SystemExit, match="max_speed .* must be no more than 500"):
        run_cli(monkeypatch, capsys, model, ["--planner", "list", "--steps", commands[0], "--max-speed", "100000"])
