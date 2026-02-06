"""Every setting in one place."""
from dataclasses import dataclass, replace


@dataclass
class Config:
    # ---- simulator ----
    img: int = 96                 # camera frame the world action model sees
    substeps: int = 10            # physics steps (2 ms) per controller tick -> 50 Hz
    ticks: int = 10               # controller ticks per action -> 5 actions a second
    max_step: float = 0.03        # metres the hand may be asked to move per action
    ctrl_clip: float = 0.02       # metres of error the controller acts on per tick
    z_hover: float = 0.12
    z_grasp: float = 0.014
    z_release: float = 0.022
    z_lifted: float = 0.05        # hand height at which a pick counts as finished
    max_actions: int = 150
    p_three_blocks: float = 0.5
    p_two_targets: float = 0.6

    # ---- controller ----
    ctrl_samples: int = 200_000
    ctrl_steps: int = 6000
    ctrl_hidden: int = 128

    # ---- demonstrations ----
    n_demos: int = 1500
    p_perturb: float = 0.2       # chance per action that the executed target is disturbed
    perturb_std: float = 0.012
    p_drop: float = 0.3           # chance per demonstration of a forced drop while carrying
    p_mistake: float = 0.04       # chance per approach action of starting a deliberate missed grasp
    workers: int = 8

    # ---- world action model ----
    chunk: int = 8                # actions generated per call (1.6 s)
    exec_h: int = 4               # actions executed before generating again
    future: tuple = (4, 8)        # which future frames are generated
    patch: int = 12
    fut_down: int = 2             # generated frames are this many times smaller than the input
    width: int = 192
    depth: int = 6
    heads: int = 6
    batch: int = 64
    lr: float = 5e-4
    wam_steps: int = 10000
    w_action: float = 10.0        # weight of the action tokens against the video tokens
    denoise_steps: int = 10
    pos_scale: float = 0.1        # metres per unit in the action chunk

    # ---- runtime ----
    subgoal_timeout: int = 45     # actions on one command before the planner is called
    surprise_steps: int = 2       # consecutive actions of "was holding, expected touch, felt none" before an event
    max_planner_calls: int = 20

    device: str = "auto"          # where the world action model trains; "auto" picks a GPU if there is one
    seed: int = 0


def quick(cfg: Config) -> Config:
    """Tiny settings that only check the plumbing."""
    return replace(cfg, ctrl_samples=4000, ctrl_steps=200, n_demos=6, workers=1, wam_steps=20,
                   width=64, depth=2, heads=4, batch=8, denoise_steps=2, max_actions=40)
