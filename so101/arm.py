"""The real SO-101 follower arm, through LeRobot 0.7.

Run on one real arm (the session in SO101.md). The automated tests cover this file with
a stand-in for LeRobot, so they do not show that LeRobot or the arm behave as expected.
"""
import time

import numpy as np

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class RealArm:
    motors = MOTORS

    def __init__(self, port, cameras, fps, ticks_per_frame, robot_id="follower", use_degrees=True,
                 width=640, height=480, dry_run=False):
        """`cameras` maps a camera name used in the dataset (for example "front") to a device index or path."""
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cams = {name: OpenCVCameraConfig(index_or_path=index, fps=fps, width=width, height=height)
                for name, index in cameras.items()}
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras=cams, use_degrees=use_degrees))
        self.fps, self.cameras, self.dry_run = fps, list(cameras), dry_run
        self.period = 1.0 / (fps * ticks_per_frame)
        self.next_tick = None
        self.late_ticks = 0          # ticks that started more than half a period behind schedule

    def connect(self):
        self.robot.connect()
        self.restart_clock()

    def restart_clock(self):
        """Start the tick schedule again from now. Called when a command begins, so the time spent
        waiting for the planner is not counted as the control loop running behind."""
        self.next_tick = time.perf_counter() + self.period

    def read_state(self):
        pos = self.robot.bus.sync_read("Present_Position", num_retry=2)
        return np.array([pos[m] for m in MOTORS], dtype=np.float64)

    def temperatures(self):
        """Every motor's temperature in degrees C, by motor name. A raw register, so it is not normalised."""
        read = self.robot.bus.sync_read("Present_Temperature", normalize=False)
        return {m: float(read[m]) for m in MOTORS}

    def read_images(self):
        return {name: self.robot.cameras[name].read_latest() for name in self.cameras}

    def send(self, target):
        if self.dry_run:
            return False
        self.robot.send_action({f"{m}.pos": float(v) for m, v in zip(MOTORS, target, strict=True)})
        return True

    def wait_tick(self):
        delay = self.next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.5 * self.period:
            self.late_ticks += 1
        self.next_tick = max(self.next_tick + self.period, time.perf_counter())

    def close(self):
        self.robot.disconnect()
