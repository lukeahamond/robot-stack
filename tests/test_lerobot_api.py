"""Checks against the real LeRobot, for the calls so101/arm.py and so101/lerobot_io.py make.

The other tests use a stand-in for LeRobot, so a LeRobot release that renames or drops something
this code calls would pass them. These run only where LeRobot is installed (the `so101` extra, Python
3.12 or later) and are skipped elsewhere. They need no arm: nothing here opens a port or a camera.
"""
import inspect

import numpy as np
import pytest

pytest.importorskip("lerobot")


def test_what_the_arm_code_calls_is_there():
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    camera = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
    config = SO101FollowerConfig(port="/dev/null", id="follower", cameras={"front": camera}, use_degrees=True)
    assert config.use_degrees is True and list(config.cameras) == ["front"]
    for name in ("connect", "disconnect", "send_action"):
        assert callable(getattr(SO101Follower, name))
    assert callable(OpenCVCamera.read_latest)
    robot = SO101Follower(config)                       # builds the bus and camera objects, opens nothing
    assert set(robot.cameras) == {"front"}
    read = inspect.signature(robot.bus.sync_read).parameters
    assert {"normalize", "num_retry"} <= set(read)      # arm.py passes both by name


def test_a_dataset_written_by_the_sink_is_read_back_by_prepare(tmp_path):
    from so101.arm import MOTORS
    from so101.data import Store
    from so101.lerobot_io import LeRobotSink, prepare
    img, cameras, lengths = 64, ["front", "wrist"], [12, 9]
    rng = np.random.default_rng(0)
    sink = LeRobotSink(tmp_path / "dataset", cameras, MOTORS, 30, img)
    episodes = []
    for n, task in zip(lengths, ["pick up the red block", "put it in the box"], strict=True):
        state = rng.uniform(-50, 50, (n, 6)).astype(np.float32)
        action = rng.uniform(-50, 50, (n, 6)).astype(np.float32)
        frames = {c: np.full((n, img, img, 3), 40 * (i + 1), dtype=np.uint8) for i, c in enumerate(cameras)}
        sink.add_episode(task, state, action, frames)
        episodes.append((task, state, action))
    sink.close()
    assert prepare([str(tmp_path / "dataset")], tmp_path / "store", img, log=lambda *_: None) == 2
    store = Store(tmp_path / "store")
    assert store.cameras == cameras and store.motors == MOTORS and store.fps == 30
    assert store.tasks == [task for task, _, _ in episodes] and store.lengths.tolist() == lengths
    assert np.allclose(store.state, np.concatenate([s for _, s, _ in episodes]))
    assert np.allclose(store.action, np.concatenate([a for _, _, a in episodes]))
    for i, c in enumerate(cameras):                     # flat grey frames survive video coding to within a few levels
        assert store.frames[c].shape == (sum(lengths), img, img, 3)
        assert abs(float(store.frames[c].mean()) - 40 * (i + 1)) < 6
