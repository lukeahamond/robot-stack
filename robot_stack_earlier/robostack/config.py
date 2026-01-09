from dataclasses import dataclass, replace


@dataclass
class Config:
    # --- simulator ---
    img: int = 32               # camera resolution (square)
    grasp_radius: float = 0.07
    bin_radius: float = 0.12
    max_step: float = 0.08      # furthest a waypoint may sit from the gripper
    max_tick: float = 0.05      # gripper speed limit per simulator tick
    substeps: int = 4           # controller ticks per action (the fast loop)
    kp: float = 0.5
    tick_noise: float = 0.004
    p_slip: float = 0.006       # chance per tick that a carried block slips
    max_actions: int = 80

    # --- planner <-> action model interface ---
    latent: int = 64
    chunk: int = 4              # actions predicted per chunk
    exec_h: int = 2             # actions executed before re-inferring
    p_drop_words: float = 0.15  # channel dropout, so neither channel is ignored
    p_drop_latent: float = 0.15

    # --- models ---
    hidden: int = 256
    ensemble: int = 4

    # --- data (the scripted expert stands in for teleoperation) ---
    n_expert: int = 2000
    n_sloppy: int = 800
    p_perturb: float = 0.12     # chance per step that an expert action is disturbed
    p_detour: float = 0.3       # chance a two-block expert episode grabs the other block first

    # --- training ---
    batch: int = 256
    lr: float = 1e-3
    steps_encoder: int = 4000
    w_reconstruct: float = 20.0  # reconstruction weight against action prediction in the encoder stage
    steps_world: int = 4000
    steps_planner: int = 3000
    steps_action: int = 5000
    steps_critic: int = 4000
    gamma: float = 0.97

    # --- runtime ---
    timer_chunks: int = 6             # planner period when there is no critic
    max_chunks_per_subgoal: int = 12  # critic-mode timeout
    done_threshold: float = 0.5
    regress_threshold: float = 0.3
    veto_fail_prob: float = 0.3
    veto_disagreement: float = 0.25
    max_vetoes: int = 3

    # --- correction rounds (expert relabels the agent's own rollouts) ---
    correction_rounds: int = 3
    correction_episodes: int = 400
    steps_refit: int = 2000

    # --- RL fine-tuning (advantage-weighted regression) ---
    rl_rounds: int = 3
    rl_episodes: int = 200
    rl_explore: float = 0.01
    rl_beta: float = 0.3
    rl_steps_action: int = 1500
    rl_steps_critic: int = 1500
    rl_steps_world: int = 1500

    device: str = "cpu"           # where everything but the encoder runs
    encoder_device: str = "auto"  # where the encoder trains; "auto" picks a GPU if there is one
    seed: int = 0


def quick(cfg: Config) -> Config:
    """Tiny settings for smoke tests: checks the plumbing, not the results."""
    return replace(
        cfg, n_expert=24, n_sloppy=24, steps_encoder=40, steps_world=40, steps_planner=40,
        steps_action=40, steps_critic=40, rl_rounds=1, rl_episodes=6, rl_steps_action=20,
        rl_steps_critic=20, rl_steps_world=20, ensemble=2, hidden=64, correction_rounds=1, correction_episodes=6,
        steps_refit=20,
    )
