"""A five-joint arm with a two-finger gripper at a table, in MuJoCo.

Two or three coloured blocks sit on the table. The instruction names one or two of them to
put on the tray. A carried block can be dropped by a forced gripper glitch, which is
recoverable. Putting a block the instruction did not name on the tray ends the episode.

One `step` is one action from the world action model (0.2 s). Inside it the fast controller
runs `ticks` times (50 Hz), and each tick advances the physics `substeps` times.
"""
import mujoco
import numpy as np

PICK, PLACE, DONE = 0, 1, 2
VERBS = ("pick", "place", "done")
COLORS = ("red", "green", "blue")
NONE = 3                                   # colour token for "no colour" (place commands)
RGBA = np.array([[0.92, 0.12, 0.12, 1], [0.1, 0.78, 0.2, 1], [0.15, 0.35, 1.0, 1]])
H0, L1, L2, L3 = 0.15, 0.30, 0.30, 0.12    # shoulder height, upper arm, forearm, wrist to fingertip
HELD_FORCE = 0.3                          # newtons on both finger pads for a block to count as held
HELD_TOUCH = float(np.tanh(HELD_FORCE / 2.0))   # the same, as a touch reading; used wherever touch is thresholded
OPEN = 0.045                               # finger travel when open (half the opening)
TRAY = np.array([0.36, 0.24])
TRAY_HALF = np.array([0.085, 0.05])
SLOTS = TRAY + np.array([[-0.04, 0.0], [0.04, 0.0]])
PROP = 6                                   # hand x, y, z, finger opening, commanded closed, touch

XML = f"""
<mujoco model="armstack">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" impratio="10" cone="elliptic" integrator="implicitfast"/>
  <visual>
    <global offwidth="512" offheight="512"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.55 0.55 0.55" specular="0 0 0"/>
  </visual>
  <default>
    <geom condim="4" friction="1.2 0.02 0.001" contype="0" conaffinity="0"/>
    <joint armature="0.02"/>
    <default class="arm"><geom rgba="0.86 0.86 0.9 1" density="300"/></default>
    <default class="pad"><geom type="box" size="0.012 0.004 0.026" rgba="0.08 0.08 0.08 1"
      density="500" friction="1.6 0.03 0.001" contype="2" conaffinity="1"/></default>
    <default class="block"><geom type="box" size="0.02 0.02 0.02" density="1000" contype="1" conaffinity="1"/></default>
  </default>
  <worldbody>
    <light pos="0.4 0 1.5" dir="0 0 -1" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <geom name="table" type="plane" size="1.5 1.5 0.1" rgba="0.2 0.2 0.23 1" contype="1" conaffinity="1"/>
    <geom name="tray" type="box" pos="{TRAY[0]} {TRAY[1]} 0.0005" size="{TRAY_HALF[0]} {TRAY_HALF[1]} 0.0005" rgba="0.9 0.8 0.3 1"/>
    <body name="lookat" pos="0.37 0.03 0.03"/>
    <camera name="front" pos="0.98 0.03 0.82" mode="targetbody" target="lookat" fovy="38"/>
    <body name="base">
      <geom class="arm" type="cylinder" size="0.045 0.075" pos="0 0 0.075" rgba="0.5 0.5 0.55 1"/>
      <body name="turret" pos="0 0 {H0}" gravcomp="1">
        <joint name="yaw" axis="0 0 1" range="-1.6 1.6" damping="8"/>
        <geom class="arm" type="sphere" size="0.032"/>
        <body name="upper" gravcomp="1">
          <joint name="shoulder" axis="0 1 0" range="-0.6 1.7" damping="8"/>
          <geom class="arm" type="capsule" fromto="0 0 0 0 0 {L1}" size="0.022"/>
          <body name="fore" pos="0 0 {L1}" gravcomp="1">
            <joint name="elbow" axis="0 1 0" range="0.1 2.8" damping="6"/>
            <geom class="arm" type="capsule" fromto="0 0 0 0 0 {L2}" size="0.018"/>
            <body name="wrist" pos="0 0 {L2}" gravcomp="1">
              <joint name="wpitch" axis="0 1 0" range="-0.8 2.8" damping="1.5"/>
              <geom class="arm" type="sphere" size="0.02"/>
              <body name="hand" gravcomp="1">
                <joint name="wyaw" axis="0 0 1" range="-1.7 1.7" damping="1"/>
                <geom class="arm" type="cylinder" fromto="0 0 0 0 0 0.052" size="0.015"/>
                <geom class="arm" name="palm" type="box" pos="0 0 0.06" size="0.016 0.058 0.008"
                      rgba="0.3 0.3 0.33 1" contype="2" conaffinity="1"/>
                <site name="ee" pos="0 0 {L3}" size="0.003" rgba="1 1 1 0"/>
                <body name="fl" pos="0 0 0.068" gravcomp="1">
                  <joint name="fl" type="slide" axis="0 1 0" range="0 {OPEN}" damping="5"/>
                  <geom class="pad" name="fl_pad" pos="0 0.004 0.026"/>
                </body>
                <body name="fr" pos="0 0 0.068" gravcomp="1">
                  <joint name="fr" type="slide" axis="0 -1 0" range="0 {OPEN}" damping="5"/>
                  <geom class="pad" name="fr_pad" pos="0 -0.004 0.026"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="block0" pos="0.3 0 0.02"><freejoint name="block0"/><geom class="block" name="block0"/></body>
    <body name="block1" pos="0.4 0 0.02"><freejoint name="block1"/><geom class="block" name="block1"/></body>
    <body name="block2" pos="0.5 0 0.02"><freejoint name="block2"/><geom class="block" name="block2"/></body>
  </worldbody>
  <actuator>
    <position name="yaw" joint="yaw" kp="300" ctrlrange="-1.6 1.6"/>
    <position name="shoulder" joint="shoulder" kp="300" ctrlrange="-0.6 1.7"/>
    <position name="elbow" joint="elbow" kp="200" ctrlrange="0.1 2.8"/>
    <position name="wpitch" joint="wpitch" kp="40" ctrlrange="-0.8 2.8"/>
    <position name="wyaw" joint="wyaw" kp="20" ctrlrange="-1.7 1.7"/>
    <position name="fl" joint="fl" kp="250" ctrlrange="0 {OPEN}" forcerange="-8 8"/>
    <position name="fr" joint="fr" kp="250" ctrlrange="0 {OPEN}" forcerange="-8 8"/>
  </actuator>
</mujoco>
"""


def fk(u):
    """Fingertip position from yaw, shoulder and elbow angles, with the hand held pointing down."""
    u = np.asarray(u, dtype=float)
    r = L1 * np.sin(u[..., 1]) + L2 * np.sin(u[..., 1] + u[..., 2])
    z = H0 + L1 * np.cos(u[..., 1]) + L2 * np.cos(u[..., 1] + u[..., 2]) - L3
    return np.stack([r * np.cos(u[..., 0]), r * np.sin(u[..., 0]), z], -1)


def ik(pos):
    """Yaw, shoulder and elbow angles that put the fingertip at `pos` (elbow up)."""
    pos = np.asarray(pos, dtype=float)
    px, pz = np.hypot(pos[..., 0], pos[..., 1]), pos[..., 2] + L3 - H0
    c = np.clip((px ** 2 + pz ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1.0, 1.0)
    q2 = np.arccos(c)
    q1 = np.arctan2(px, pz) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.stack([np.arctan2(pos[..., 1], pos[..., 0]), q1, q2], -1)


class ArmEnv:
    def __init__(self, cfg, controller, seed=0):
        self.cfg, self.controller = cfg, controller
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.ee_site = m.site("ee").id
        self.pads = (m.geom("fl_pad").id, m.geom("fr_pad").id)
        self.block_geom = [m.geom(f"block{i}").id for i in range(3)]
        self.block_qadr = [m.joint(f"block{i}").qposadr[0] for i in range(3)]
        self.block_body = [m.body(f"block{i}").id for i in range(3)]
        self.cam = m.camera("front").id
        self._renderers = {}
        self.reset(seed)

    # ---- episode setup --------------------------------------------------------
    def reset(self, seed=None, drop=False):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r, c, m, d = self.rng, self.cfg, self.model, self.data
        mujoco.mj_resetData(m, d)
        self.n = 3 if r.random() < c.p_three_blocks else 2
        self.color = r.permutation(3)                       # colour of block i
        n_targets = 2 if r.random() < c.p_two_targets else 1
        self.targets = [int(i) for i in r.permutation(self.n)[:n_targets]]   # block indices, in order
        while True:
            xy = r.uniform([0.26, -0.24], [0.46, 0.10], size=(self.n, 2))
            gaps = [np.linalg.norm(xy[i] - xy[j]) for i in range(self.n) for j in range(i)]
            if min(gaps) > 0.10:
                break
        for i in range(3):
            a = self.block_qadr[i]
            d.qpos[a:a + 3] = [xy[i, 0], xy[i, 1], 0.02] if i < self.n else [-1.0, 0.3 * i, 0.02]
            d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            m.geom_rgba[self.block_geom[i]] = RGBA[self.color[i]]
        start = r.uniform([0.30, -0.10, 0.12], [0.42, 0.10, 0.16])
        u = ik(start)
        d.qpos[:5] = [u[0], u[1], u[2], np.pi - u[1] - u[2], u[0]]
        d.qpos[5:7] = OPEN
        d.ctrl[:5], d.ctrl[5:7] = d.qpos[:5], OPEN
        mujoco.mj_forward(m, d)
        self.grip_closed = False
        self.t = 0
        self.touch, self.held = 0.0, -1
        self.wrong = False
        self.drop_at = int(r.integers(2, 7)) if drop else -1   # carried actions before the forced drop
        self._carried, self._glitch = 0, 0
        self.dropped = False
        return self

    # ---- one action -----------------------------------------------------------
    def step(self, target, grip_close):
        c, m, d = self.cfg, self.model, self.data
        if self._glitch > 0:
            grip_close, self._glitch = False, self._glitch - 1
        self.grip_closed = bool(grip_close)
        target = np.asarray(target, dtype=float)
        for _ in range(c.ticks):
            self.set_arm(self.controller.tick(self.arm_q, self.ee, target))
            d.ctrl[5:7] = 0.0 if grip_close else OPEN
            mujoco.mj_step(m, d, c.substeps)
        self._sense()
        self.t += 1
        if self.held >= 0 and self.block_pos(self.held)[2] > 0.08 and not self._over_tray(self.held):
            self._carried += 1
            if self._carried == self.drop_at:
                self._glitch, self.dropped = 2, True
        for i in range(self.n):
            if i not in self.targets and self.in_tray(i):
                self.wrong = True

    def set_arm(self, u):
        self.data.ctrl[:5] = [u[0], u[1], u[2], np.pi - u[1] - u[2], u[0]]

    def _sense(self):
        """Touch: the smaller of the two finger-pad forces on a block, squashed to 0..1."""
        d = self.data
        force = np.zeros((3, 2))
        f6 = np.zeros(6)
        for k in range(d.ncon):
            con = d.contact[k]
            for g_pad, g_block in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                if g_pad in self.pads and g_block in self.block_geom:
                    mujoco.mj_contactForce(self.model, d, k, f6)
                    force[self.block_geom.index(g_block), self.pads.index(g_pad)] += abs(f6[0])
        grip = force.min(1)
        best = int(grip.argmax())
        self.touch = float(np.tanh(grip[best] / 2.0))
        self.held = best if grip[best] > HELD_FORCE else -1

    # ---- state the robot can sense ------------------------------------------
    @property
    def arm_q(self):
        return self.data.qpos[:3].copy()

    @property
    def ee(self):
        return self.data.site_xpos[self.ee_site].copy()

    def proprio(self):
        d = self.data
        return np.array([*self.ee, d.qpos[5] + d.qpos[6], float(self.grip_closed), self.touch], dtype=np.float32)

    def frame(self, size=None):
        """Camera image, uint8, (size, size, 3). Rendered at twice the size and averaged down."""
        size = size or self.cfg.img
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, 2 * size, 2 * size)
        ren = self._renderers[size]
        ren.update_scene(self.data, camera=self.cam)
        big = ren.render().astype(np.float32)
        return big.reshape(size, 2, size, 2, 3).mean((1, 3)).round().astype(np.uint8)

    def close(self):
        """Free the renderers' GL contexts. The environment can still be used; they are made again when needed."""
        for ren in getattr(self, "_renderers", {}).values():
            ren.close()
        self._renderers = {}

    # ---- task (simulator truth; used for labels, scripted planners and scoring) ----
    def block_pos(self, i):
        return self.data.xpos[self.block_body[i]].copy()

    def _over_tray(self, i):
        return bool((np.abs(self.block_pos(i)[:2] - TRAY) < TRAY_HALF).all())

    def in_tray(self, i):
        return self._over_tray(i) and self.block_pos(i)[2] < 0.035 and self.held != i

    def next_slot(self):
        for s in SLOTS:
            if all(np.linalg.norm(self.block_pos(i)[:2] - s) > 0.045 or self.held == i for i in range(self.n)):
                return s
        return SLOTS[0]

    def oracle_subgoal(self):
        for i in self.targets:
            if not self.in_tray(i):
                if self.held == i and (self.block_pos(i)[2] > 0.05 or self._over_tray(i)):
                    return (PLACE, NONE)
                return (PICK, int(self.color[i]))
        return (DONE, NONE)

    def block_of_color(self, color):
        """The block of this colour. A colour not on the table is an error: -1 would read as
        "holding nothing" (`held == -1`) and as the last block (`block_pos(-1)`)."""
        for i in range(self.n):
            if self.color[i] == color:
                return i
        raise ValueError(f"no block of colour {color} on the table")

    @property
    def instruction(self):
        names = [COLORS[self.color[i]] for i in self.targets]
        text = f"Put the {names[0]} block on the tray"
        return text + (f", then the {names[1]} block." if len(names) > 1 else ".")

    @property
    def success(self):
        return all(self.in_tray(i) for i in self.targets) and not self.wrong

    @property
    def done(self):
        return self.wrong or self.success


def subgoal_text(sg):
    verb, color = int(sg[0]), int(sg[1])
    if verb == DONE:
        return "done"
    return f"pick up the {COLORS[color]} block" if verb == PICK else "put it on the tray"
