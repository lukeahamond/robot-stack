"""A simulated stand-in for the SO-101, for testing the software without the arm.

It is not a model of the real SO-101. It is the five-joint arm from `robot_stack_earlier.armstack`, given the
SO-101's joint names, a front camera and a wrist camera, joint values in degrees and a
gripper that reads 0 (closed) to 100 (open). It answers the same calls as the real arm, so
recording, training and running go through the same code.

Also here: a scripted demonstrator that stands in for a person with the leader arm, and
the scripted planners used for scoring.
"""
import mujoco
import numpy as np

from robot_stack_earlier.armstack.config import Config as ArmConfig
from robot_stack_earlier.armstack.control import clip_norm
from robot_stack_earlier.armstack.expert import funnel
from robot_stack_earlier.armstack.sim import DONE as SG_DONE
from robot_stack_earlier.armstack.sim import OPEN, PICK, XML, ArmEnv, ik, subgoal_text

from .planner import DONE

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
WRIST_CAM = '<camera name="wrist" pos="0.06 0 0.01" xyaxes="0 1 0 0.909 0 0.417" fovy="75"/>\n                <site name="ee"'
SUBSTEPS = 10            # physics steps per controller tick: 50 ticks a second
LEAD = 0.006             # how far ahead of the fingertip the demonstrator aims, metres


class SimArm(ArmEnv):
    fps = 25
    cameras = ["front", "wrist"]
    motors = MOTORS

    def __init__(self, img=112, seed=0):
        self.cfg, self.controller, self.img = ArmConfig(), None, img
        self.model = mujoco.MjModel.from_xml_string(XML.replace('<site name="ee"', WRIST_CAM))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.ee_site = m.site("ee").id
        self.pads = (m.geom("fl_pad").id, m.geom("fr_pad").id)
        self.block_geom = [m.geom(f"block{i}").id for i in range(3)]
        self.block_qadr = [m.joint(f"block{i}").qposadr[0] for i in range(3)]
        self.block_body = [m.body(f"block{i}").id for i in range(3)]
        self.cam = m.camera("front").id
        self._renderer = None
        self.reset(seed)

    def reset(self, seed=None, drop=False):
        super().reset(seed, False)
        self.drop_after = int(self.rng.integers(20, 70)) if drop else -1    # ticks of carrying before the forced drop
        self.carried = self.glitch = 0
        self.dropped = False
        return self

    # ---- what the real arm also provides ---------------------------------------
    def read_state(self):
        q = self.data.qpos
        return np.array([*np.degrees(q[:5]), 100.0 * (q[5] + q[6]) / (2 * OPEN)])

    def read_images(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 2 * self.img, 2 * self.img)
        out = {}
        for cam in self.cameras:
            self._renderer.update_scene(self.data, camera=cam)
            big = self._renderer.render().astype(np.float32)
            out[cam] = big.reshape(self.img, 2, self.img, 2, 3).mean((1, 3)).round().astype(np.uint8)
        return out

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    def send(self, target):
        self.data.ctrl[:5] = np.radians(target[:5])
        grip = OPEN if self.glitch > 0 else np.clip(target[5], 0.0, 100.0) / 100.0 * OPEN
        self.data.ctrl[5:7] = grip

    def wait_tick(self):
        mujoco.mj_step(self.model, self.data, SUBSTEPS)
        self._sense()
        self.glitch = max(0, self.glitch - 1)
        if self.held >= 0 and self.block_pos(self.held)[2] > 0.08 and not self._over_tray(self.held):
            self.carried += 1
            if self.carried == self.drop_after:
                self.glitch, self.dropped = 20, True
        for i in range(self.n):
            if i not in self.targets and self.in_tray(i):
                self.wrong = True


# ---- scripted demonstrator ---------------------------------------------------------
def joints_for(point, grip):
    u = ik(point)
    return np.array([*np.degrees([u[0], u[1], u[2], np.pi - u[1] - u[2], u[0]]), grip])


def demo_step(arm, verb, color):
    """What a person with the leader arm would command next. -> (six joint targets, finished)."""
    c, ee = arm.cfg, arm.ee
    if verb == PICK:
        i = arm.block_of_color(color)
        if arm.held == i:
            if ee[2] >= c.z_lifted + 0.02:
                return joints_for(ee, 0.0), True
            return joints_for(ee + clip_norm(np.array([ee[0], ee[1], c.z_hover]) - ee, LEAD), 0.0), False
        if arm.held >= 0:            # holding another block: set it down where it is, then go for the right one
            if ee[2] > c.z_grasp + 0.01:
                return joints_for(ee + clip_norm(np.array([ee[0], ee[1], c.z_grasp]) - ee, LEAD), 0.0), False
            return joints_for(ee, 100.0), False
        goal, d = funnel(c, ee, arm.block_pos(i)[:2], c.z_grasp)
        close = d < 0.012 and ee[2] < c.z_grasp + 0.008
        return joints_for(ee + clip_norm(goal - ee, LEAD), 0.0 if close else 100.0), False
    if arm.held >= 0:
        goal, d = funnel(c, ee, arm.next_slot(), c.z_release)
        let_go = d < 0.012 and ee[2] < c.z_release + 0.008
        return joints_for(ee + clip_norm(goal - ee, LEAD), 100.0 if let_go else 0.0), False
    if ee[2] >= c.z_hover - 0.035:
        return joints_for(ee, 100.0), True
    return joints_for(ee + clip_norm(np.array([ee[0], ee[1], c.z_hover - 0.02]) - ee, LEAD), 100.0), False


def frame_of(arm, action, ticks_per_frame=2):
    for _ in range(ticks_per_frame):
        arm.send(action)
        arm.wait_tick()


def wander(arm, rng, verb, color):
    """Unrecorded motion before an episode, so that episodes start from awkward places too."""
    if verb == PICK:
        target = arm.block_of_color(color)
        others = [i for i in range(arm.n) if i != target and not arm.in_tray(i)]
        if others and rng.random() < 0.4:
            # Start with the wrong block in the hand, so the recording shows setting it down first.
            wrong = int(arm.color[others[int(rng.integers(len(others)))]])
            for _ in range(250):
                action, done = demo_step(arm, PICK, wrong)
                frame_of(arm, action)
                if done:
                    break
            return
        xy = arm.block_pos(target)[:2]
        r, a = rng.uniform(0.03, 0.09), rng.uniform(0, 2 * np.pi)
        point = np.array([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), rng.uniform(0.02, 0.12)])
        grip = 0.0 if rng.random() < 0.5 else 100.0
    else:
        point = np.array([rng.uniform(0.28, 0.44), rng.uniform(-0.2, 0.2), rng.uniform(0.07, 0.14)])
        grip = 0.0
    for _ in range(40):
        ee = arm.ee
        frame_of(arm, joints_for(ee + clip_norm(point - ee, LEAD), grip))


def record_command(arm, verb, color, tail, max_frames=300):
    """Run the demonstrator for one command. -> (state, action, frames, succeeded)."""
    states, actions, frames = [], [], {c: [] for c in arm.cameras}
    i = arm.block_of_color(color) if verb == PICK else arm.held
    after = 0
    for _ in range(max_frames):
        state, images = arm.read_state(), arm.read_images()
        action, finished = demo_step(arm, verb, color)
        states.append(state)
        actions.append(action)
        for c in arm.cameras:
            frames[c].append(images[c])
        frame_of(arm, action)
        after = after + 1 if finished else 0
        if after >= tail + 2:
            break
    ok = after > 0 and not arm.wrong and (arm.held == i if verb == PICK else (i >= 0 and arm.in_tray(i)))
    return np.array(states), np.array(actions), {c: np.stack(v) for c, v in frames.items()}, ok


def record(sink, n_tasks, img=112, tail=8, seed0=0, p_messy=0.35, p_drop=0.25, log=print):
    """Record demonstrations of `n_tasks` whole tasks, one dataset episode per command.

    `sink.add_episode(task, state, action, frames)` receives each kept episode. Episodes
    where the command did not succeed are thrown away, as a person would re-record them.
    """
    arm = SimArm(img)
    try:
        return _record(arm, sink, n_tasks, tail, seed0, p_messy, p_drop, log)
    finally:
        arm.close()


def _record(arm, sink, n_tasks, tail, seed0, p_messy, p_drop, log):
    rng = np.random.default_rng(seed0)
    kept = thrown = 0
    for t in range(n_tasks):
        arm.reset(seed0 + t, drop=rng.random() < p_drop)
        for _ in range(8):
            verb, color = arm.oracle_subgoal()
            if verb == SG_DONE or arm.wrong:
                break
            if rng.random() < p_messy:
                wander(arm, rng, verb, color)
                verb, color = arm.oracle_subgoal()
                if verb == SG_DONE or arm.wrong:
                    break
            state, action, frames, ok = record_command(arm, verb, color, tail)
            if ok:
                sink.add_episode(subgoal_text((verb, color)), state, action, frames)
                kept += 1
            else:
                thrown += 1
        if (t + 1) % max(1, n_tasks // 10) == 0:
            log(f"recorded {t + 1}/{n_tasks} tasks: {kept} episodes kept, {thrown} thrown away")
    return kept, thrown


# ---- scripted planners, for scoring ------------------------------------------------
class SeeingPlanner:
    """Reads the simulator's true state. Stands in for a planner that looks at the cameras."""

    def __init__(self, arm):
        self.arm = arm
        self.last_observation = ""

    def next(self, instruction, commands, history, images):
        sg = self.arm.oracle_subgoal()
        return DONE if sg[0] == SG_DONE else subgoal_text(sg)


def fixed_list(arm):
    """The commands a planner would write down in advance, without looking again."""
    steps = []
    for i in arm.targets:
        steps += [subgoal_text((PICK, int(arm.color[i]))), "put it on the tray"]
    return steps
