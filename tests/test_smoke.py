"""Fast checks of the simulator's rules and of the pipeline's plumbing.

These do not check that the models learn anything: the quick config trains for a few dozen
steps. Run `python3 -m robot_stack_earlier.robostack.cli all` for real numbers.
"""
from dataclasses import replace

import numpy as np
import torch

from robot_stack_earlier.robostack.config import Config, quick
from robot_stack_earlier.robostack.controller import Controller
from robot_stack_earlier.robostack.data import CORRECTED, EXPERT, POLICY, SLOPPY, Buffer, collect, collect_episode
from robot_stack_earlier.robostack.env import BIN_XY, PICK, PROP, ToyKitchen
from robot_stack_earlier.robostack.expert import expert_action, oracle_subgoal
from robot_stack_earlier.robostack.runtime import VARIANTS, Agent, evaluate
from robot_stack_earlier.robostack.train import bc_probs, rl_finetune, train_all


def test_expert_solves_the_task_and_recovers_from_slips():
    cfg = Config()
    eps = [collect_episode(cfg, seed, EXPERT) for seed in range(40)]
    assert np.mean([e["success"] for e in eps]) >= 0.95


def test_wrong_bin_is_irreversible_and_right_bin_succeeds():
    cfg = replace(Config(), p_slip=0.0, tick_noise=0.0)
    for wrong in (False, True):
        env = ToyKitchen(cfg, seed=3)
        env.n, env.block_color, env.target_bin_color = 1, env.block_color[:1], env.target_bin_color[:1]
        env.block_xy, env.block_bin = env.block_xy[:1], env.block_bin[:1]
        right_slot = int(np.where(env.bin_color == env.target_bin_color[0])[0][0])
        slot = (right_slot + 1) % 3 if wrong else right_slot
        env.grip_xy = env.block_xy[0].copy()
        env.tick([0, 0], True)
        assert env.holding == 0 and env.achieved()[PICK, env.block_color[0]]
        env.grip_xy = BIN_XY[slot].copy()
        env.tick([0, 0], False)
        assert env.failed == wrong and env.success == (not wrong) and env.done


def test_release_over_the_table_is_recoverable():
    cfg = replace(Config(), p_slip=0.0, tick_noise=0.0)
    env = ToyKitchen(cfg, seed=5)
    env.grip_xy = env.block_xy[0].copy()
    env.tick([0, 0], True)
    env.grip_xy = np.array([0.5, 0.4])
    env.tick([0, 0], False)
    assert env.holding == -1 and not env.done and env.block_bin[0] == -1
    assert oracle_subgoal(env)[0] == PICK


def test_observation_shapes():
    cfg = Config()
    env = ToyKitchen(cfg, seed=0)
    assert env.render().shape == (3, cfg.img, cfg.img)
    assert env.proprio().shape == (PROP,)
    assert env.instruction.shape == (4,)


def test_buffer_labels_are_consistent():
    cfg = Config()
    eps = [collect_episode(cfg, s, EXPERT) for s in range(10)] + [collect_episode(cfg, 100 + s, SLOPPY) for s in range(10)]
    buf = Buffer(cfg, eps).build()
    F, S = buf.F, buf.S
    assert len(F["frames"]) == len(S["fi"]) + len(eps)          # one extra frame per episode
    assert S["chunk"].shape[1:] == (cfg.chunk, 3)
    assert (S["goal_fi"] > S["fi"]).all()                       # the goal frame lies in the future
    assert ((F["ret"] >= 0) & (F["ret"] <= 1)).all()
    # failures only ever happen in the deliberately sloppy episodes here
    assert not S["fail"][S["mode"] == EXPERT].any()
    assert S["fail"].any()
    # expert labels and executed actions differ only where the expert was perturbed
    same = (S["chunk"][:, 0] == S["executed"]).all(-1)
    assert same[S["mode"] == SLOPPY].all() and not same[S["mode"] == EXPERT].all()


def test_pipeline_runs_end_to_end(tmp_path):
    cfg = quick(Config())
    torch.manual_seed(0)
    models, buf = train_all(cfg, collect(cfg, log=lambda *_: None), log=lambda *_: None)
    assert (buf.S["mode"] == CORRECTED).any()
    assert bc_probs(buf).sum() > 0
    assert not any(p.requires_grad for p in models.encoder.parameters())   # the shared space is frozen

    n_before = len(buf.eps)
    rl_finetune(cfg, models, buf, log=lambda *_: None)
    assert len(buf.eps) == n_before + cfg.rl_rounds * cfg.rl_episodes
    assert (buf.S["mode"] == POLICY).any()

    results = evaluate(cfg, models, n=3)
    assert set(results) == set(VARIANTS)
    for r in results.values():
        assert abs(r["success"] + r["irreversible_failure"] + r["timeout"] - 1) < 1e-6


def test_event_driven_planner_is_called_less_than_every_chunk():
    """Structural property of the runtime, independent of model quality."""
    cfg = quick(Config())
    torch.manual_seed(0)
    models, _ = train_all(cfg, collect(cfg, log=lambda *_: None), log=lambda *_: None)
    every = Agent(cfg, models, critic=False, rollout_check=False, timer=1).run(ToyKitchen(cfg, 1))[0]
    timed = Agent(cfg, models, critic=False, rollout_check=False).run(ToyKitchen(cfg, 1))[0]
    # Both episodes must run their full length, or the counts are not comparable and nothing is shown.
    assert every["actions"] == timed["actions"] == cfg.max_actions
    assert timed["planner_calls"] < every["planner_calls"]


def test_controller_reaches_a_waypoint():
    cfg = replace(Config(), tick_noise=0.0)
    env = ToyKitchen(cfg, seed=0)
    target = env.grip_xy + np.array([0.05, -0.05])
    Controller(cfg).track(env, target, False)
    assert np.linalg.norm(env.grip_xy - target) < 0.01
    a, _ = expert_action(env, np.random.default_rng(0))
    assert a.shape == (3,) and np.linalg.norm(a[:2] - env.grip_xy) <= cfg.max_step + 0.03


def test_early_stops_are_not_counted_as_timeouts():
    from robot_stack_earlier.robostack.runtime import format_table
    row = dict(success=0.5, irreversible_failure=0.1, timeout=0.2, stopped_early=0.2, planner_calls=3, vetoes=0, actions=40)
    assert "stopped early" in format_table({"v": row})


def test_eval_refuses_flags_it_would_ignore(tmp_path):
    import pytest
    from robot_stack_earlier.robostack.cli import main
    with pytest.raises(SystemExit):
        main(["eval", "--run", str(tmp_path), "--seed", "3"])


def test_seed_blocks_do_not_overlap_each_other_or_the_evaluation():
    import pytest
    from robot_stack_earlier.robostack.data import seed_base
    c0, c1 = quick(Config()), quick(Config(seed=1))
    top0 = seed_base(c0) + 2_000_000 + c0.rl_rounds * 100_000          # highest training seed for seed 0
    assert top0 < 9_000_000 <= 9_000_000 + 500 < seed_base(c1)
    with pytest.raises(ValueError):
        seed_base(Config(seed=-1))


def test_episode_count_is_checked(tmp_path):
    import pytest
    from robot_stack_earlier.robostack.cli import main
    with pytest.raises(SystemExit):
        main(["all", "--run", str(tmp_path), "--episodes", "0"])
