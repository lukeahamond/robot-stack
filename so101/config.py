"""Every setting for the SO-101 stack, in one place.

Lengths of time are in dataset frames unless the name says seconds. Joint values are in
whatever units the dataset was recorded in. LeRobot 0.7's SO-101 driver gives degrees for the
five joints by default (`use_degrees`, on unless turned off) and 0..100 for the gripper; with
`use_degrees` off the joints are -100..100.
"""
from dataclasses import dataclass, fields, replace

MAX_SPEED_CEILING = 500.0   # `max_speed` above this is refused: more than three times the shipped limit, it is no limit


@dataclass(frozen=True)
class Config:
    # ---- data ---------------------------------------------------------------
    img: int = 224                # camera frames are resized to img x img; must be a multiple of 14
    tail: int = 8                 # the last frames of an episode count as "command finished"
    touch_lag: int = 6            # the gripper is compared with this many recent commands
    touch_full: float = 8.0       # gripper held this far open of every recent command = full touch
    val_frac: float = 0.05        # share of episodes held out

    # ---- world action model -------------------------------------------------
    encoder: str = "dinov2_small"  # "dinov2_small", "dinov2_base", or "tiny" (tests only)
    unfreeze: int = 2             # last encoder blocks that are fine-tuned; the rest stay frozen
    grid: int = 8                 # image tokens per side, per camera, after pooling
    fut_grid: int = 4             # predicted-future tokens per side, per camera
    chunk: int = 30               # frames of movement generated per call (1 s at 30 fps)
    future: tuple = (15, 30)      # how far ahead the predicted camera features are, in frames
    width: int = 512
    enc_depth: int = 4            # layers that read the cameras, body state and command, once per call
    dec_depth: int = 6            # layers that denoise the future and the movement
    heads: int = 8
    p_drop_future: float = 0.3    # share of training examples with the future tokens hidden,
                                  # so the model also works with future prediction switched off
    denoise_steps: int = 8

    # ---- training -----------------------------------------------------------
    batch: int = 64
    steps: int = 20000
    lr: float = 3e-4
    enc_lr: float = 2e-5
    warmup: int = 500
    w_action: float = 10.0
    w_future: float = 1.0
    p_tail: float = 0.1           # share of each batch drawn from the last frames of episodes
    shift: int = 8                # random image shift in pixels (at img = 224)
    amp: bool = True              # 16-bit floats on NVIDIA GPUs
    seed: int = 0
    device: str = "auto"
    prefetch: int = 2             # training batches made ahead on another thread while the GPU trains; 0 = off

    # ---- controller ---------------------------------------------------------
    ticks_per_frame: int = 2      # controller ticks per dataset frame: 60 Hz for 30 fps data
    execute: int = 12             # frames of a chunk that run before a new chunk is asked for
    blend: int = 4                # frames over which a new chunk takes over from the old one
    max_speed: float = 150.0      # fastest any joint may be commanded to move, units per second
    limit_margin: float = 5.0     # joints may go this far past the range seen in the data
    grip_squeeze: float = 25.0    # the gripper is never commanded further closed than this past where it is
    max_lead: float = 30.0        # a joint is not aimed further than this from where it is, so a blocked
                                  # joint is not pushed harder and harder. The speed limit comes after it:
                                  # if the arm is pushed away suddenly, the target follows at max_speed
                                  # rather than jumping, and is further than this until it catches up
    block_time: float = 0.3       # seconds a joint may lag its target by more than `idle_lead` without moving
                                  # `idle_lead` towards it. After that it counts as blocked (held by hand, or
                                  # against something) and is aimed no further than `idle_lead` from where it
                                  # is, so letting it go moves it at most `idle_lead` at the servo's own speed
                                  # instead of up to `max_lead`. It counts as free again once it moves
                                  # `idle_lead` towards its target, or its target comes within `idle_lead`
    idle_squeeze: float = 12.0    # while nothing is running (waiting for the planner, or for Enter at the end) the
                                  # gripper squeezes only this far past where it is, so a stalled gripper motor
                                  # does not heat up. Above touch_full, so a held block still reads as full touch
    idle_lead: float = 5.0        # while nothing is running, a joint is aimed no further than this from where it
                                  # is, so a joint held against an obstacle does not push at up to `max_lead` for
                                  # a whole planner call. Wide enough that an arm sagging a little under its own
                                  # weight keeps its target
    grip_hot: float = 60.0        # gripper motor temperature, degrees C, at which the run ends holding still
                                  # (the real arm only; read after each planner answer and after each command)
    joint_hot: float = 65.0       # the same for the five joint motors

    # ---- events sent up to the planner ----------------------------------------
    hold_time: float = 0.2        # seconds of touch before the hand counts as holding something
    hold_still: float = 20.0      # ...and only while the gripper moves slower than this, units per second:
                                  # a gripper closing on nothing also reads as touch until it stops
    surprise_time: float = 0.3    # seconds of "expected touch, felt none" before the alarm fires
    finish_votes: int = 2         # chunks in a row that must say "finished"
    command_timeout: float = 30.0  # seconds one command may run
    max_commands: int = 20        # commands the planner may give for one instruction

    # ---- planner ------------------------------------------------------------
    planner_model: str = "claude-opus-5-5"
    planner_effort: str = "low"
    planner_fallbacks: bool = True  # let the API hand a declined request to a fallback model
    planner_image: int = 512      # longest side of the pictures sent to the planner
    planner_timeout: float = 60.0  # seconds one planner call may take, retries included, before the arm gives up and holds still

    def __post_init__(self):
        """Refuse settings that would make the arm unsafe or quietly turn a safeguard off.

        The settings cannot be changed afterwards, so nothing gets past this check: make a changed
        copy with `dataclasses.replace`, which runs it again.
        """
        bad = []
        for name in ("max_speed", "max_lead", "touch_full", "hold_time", "surprise_time", "command_timeout",
                     "planner_timeout", "grip_hot", "joint_hot", "hold_still", "idle_lead", "block_time"):
            if not getattr(self, name) > 0:
                bad.append(f"{name} must be above 0 (it is {getattr(self, name)!r})")
        if self.prefetch < 0:
            bad.append(f"prefetch must not be below 0 (it is {self.prefetch!r})")
        if self.shift < 0:
            bad.append(f"shift must not be below 0 (it is {self.shift!r})")
        if self.limit_margin < 0:
            bad.append(f"limit_margin must not be below 0 (it is {self.limit_margin!r})")
        if self.max_speed > MAX_SPEED_CEILING:
            bad.append(f"max_speed ({self.max_speed!r}) must be no more than {MAX_SPEED_CEILING:g}, or it limits nothing")
        if self.idle_lead > self.max_lead:
            bad.append(f"idle_lead ({self.idle_lead}) must be no more than max_lead ({self.max_lead})")
        if self.img < 14 or self.img % 14:
            bad.append(f"img must be a multiple of 14, the encoder's patch size (it is {self.img!r})")
        # touch_lag 0 makes touch always read nothing, so the drop alarm never arms. denoise_steps 0 sends
        # the arm the noise the model starts from. tail 0 leaves no frame marked "finished". warmup 0
        # divides by zero in the learning-rate schedule.
        for name in ("ticks_per_frame", "execute", "chunk", "steps", "batch", "finish_votes", "max_commands",
                     "touch_lag", "tail", "denoise_steps", "planner_image", "warmup"):
            if not getattr(self, name) >= 1:
                bad.append(f"{name} must be 1 or more (it is {getattr(self, name)!r})")
        if self.execute >= self.chunk:
            bad.append(f"execute ({self.execute}) must be below chunk ({self.chunk}), or the arm stops at the end "
                       f"of every chunk before the next is asked for")
        if not 0 <= self.blend <= self.execute:
            bad.append(f"blend ({self.blend}) must be between 0 and execute ({self.execute})")
        # Touch is the gap between where the gripper is and what it was told, and the squeeze limit caps that
        # gap. At or below touch_full a held block can never read as full touch, and the drop alarm never arms.
        if not self.grip_squeeze > self.touch_full:
            bad.append(f"grip_squeeze ({self.grip_squeeze}) must be above touch_full ({self.touch_full}), "
                       f"or a held block never reads as touch and the drop alarm is off")
        if not self.touch_full < self.idle_squeeze <= self.grip_squeeze:
            bad.append(f"idle_squeeze ({self.idle_squeeze}) must be above touch_full ({self.touch_full}) "
                       f"and no more than grip_squeeze ({self.grip_squeeze})")
        if bad:
            raise ValueError("bad settings: " + "; ".join(bad))


# Settings a trained model depends on. They are read back from the saved model. Every other
# setting (controller, events, planner) applies when the arm runs and can be changed without
# training again.
TRAINED = ("img", "tail", "touch_lag", "touch_full", "encoder", "unfreeze", "grid", "fut_grid", "chunk",
           "future", "width", "enc_depth", "dec_depth", "heads")


def from_saved(saved, settings=None):
    """The settings to use with a saved model. `saved` is the dictionary stored with the weights.

    Without `settings`, everything is as it was when the model was trained. With `settings`,
    only the `TRAINED` fields come from the saved model and the rest come from `settings`.
    Fields this version does not have are ignored, so older saved models still load.
    """
    known = {f.name for f in fields(Config)}
    saved = {k: v for k, v in saved.items() if k in known}
    if "future" in saved:
        saved["future"] = tuple(saved["future"])
    if settings is None:
        return Config(**saved)
    return replace(settings, **{k: saved[k] for k in TRAINED if k in saved})


def sim(cfg=None):
    """Settings for the simulated stand-in arm (25 fps, small pictures, laptop-sized model)."""
    return replace(cfg or Config(), img=112, chunk=24, future=(12, 24), execute=8, blend=3,
                   width=256, enc_depth=3, dec_depth=4, heads=8, batch=32, steps=6000,
                   shift=4, command_timeout=14.0)


def quick(cfg=None):
    """Tiny settings that only check the plumbing."""
    return replace(sim(cfg), img=56, encoder="tiny", unfreeze=0, grid=4, fut_grid=2, width=64,
                   enc_depth=1, dec_depth=1, heads=4, batch=8, steps=30, warmup=5, denoise_steps=2)
