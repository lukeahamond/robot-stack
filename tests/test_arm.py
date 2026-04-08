"""Fast checks of the arm simulator and of the pipeline's plumbing. They do not check
that the world action model learns anything; run `python3 -m robot_stack_earlier.armstack.cli train` for that."""
from dataclasses import replace

import numpy as np
import pytest

from robot_stack_earlier.armstack.config import Config, quick
from robot_stack_earlier.armstack.control import AnalyticController, LearnedController, train_controller
from robot_stack_earlier.armstack.data import Dataset, collect
from robot_stack_earlier.armstack.expert import Expert
from robot_stack_earlier.armstack.runtime import PLANNERS, Stack, parse_command, run_episode
from robot_stack_earlier.armstack.sim import NONE, PICK, PLACE, ArmEnv, fk, ik
from robot_stack_earlier.armstack.train import train_wam


def test_arm_geometry_matches_the_simulator():
    pos = np.array([0.41, -0.13, 0.06])
    assert np.abs(fk(ik(pos)) - pos).max() < 1e-9
    env = ArmEnv(Config(), AnalyticController(Config()), seed=2)
    assert np.abs(env.ee - fk(env.arm_q)).max() < 1e-6


def test_demonstrator_solves_the_task_including_after_a_drop():
    cfg = Config()
    env = ArmEnv(cfg, AnalyticController(cfg))
    wins, drops = [], 0
    for seed in range(12):
        env.reset(seed, drop=seed % 2 == 0)
        expert = Expert(cfg)
        while env.t < cfg.max_actions and not env.done:
            target, grip, _ = expert.act(env)
            env.step(target, grip)
        wins.append(env.success)
        drops += env.dropped
    assert np.mean(wins) >= 0.9 and drops >= 4


def test_learned_controller_tracks_a_target():
    cfg = replace(Config(), ctrl_samples=20000, ctrl_steps=800)
    env = ArmEnv(cfg, LearnedController(cfg, train_controller(cfg, log=lambda *_: None)), seed=1)
    target = np.array([0.44, -0.18, 0.05])
    for _ in range(14):
        env.step(env.ee + np.clip(target - env.ee, -0.03, 0.03), False)
    assert np.linalg.norm(env.ee - target) < 0.003


def test_pipeline_runs_end_to_end():
    cfg = quick(Config())
    ctrl = train_controller(cfg, log=lambda *_: None)
    ds = Dataset(cfg, collect(cfg, ctrl, log=lambda *_: None))
    assert ds.chunk.shape[1:] == (cfg.chunk, 6)
    assert (ds.chunk[:, :, 5].diff(dim=1) >= 0).all()          # once a command is finished it stays finished
    assert (ds.fut >= ds.fi[:, None]).all()                     # generated frames lie in the future
    assert (ds.chunk[:, :, :3].diff(dim=1).norm(dim=-1) <= cfg.max_step / cfg.pos_scale + 1e-4).all()   # no step longer than allowed
    model = train_wam(cfg, ds, log=lambda *_: None, device="cpu")
    frames, act = model.sample(*ds.batch(ds.train_idx[:2], "cpu")[:3])
    assert frames.shape == (2, len(cfg.future), cfg.img // cfg.fut_down, cfg.img // cfg.fut_down, 3)
    assert act.shape == (2, cfg.chunk, 6)
    stack = Stack(cfg, ctrl, model)
    for make in PLANNERS.values():
        r = run_episode(stack, make, seed=1, drop=True)
        assert r["success"] + r["wrong"] + r["unfinished"] == 1
    # the same seed replays to the same state, which the by-hand planner tool relies on
    a = stack.reset(5).run_command((PICK, 0)), stack.env.data.qpos.copy()
    b = stack.reset(5).run_command((PICK, 0)), stack.env.data.qpos.copy()
    assert a[0] == b[0] and np.allclose(a[1], b[1])


def test_commands_parse_to_tokens():
    assert parse_command("Pick up the blue block") == (PICK, 2)
    assert parse_command("put it on the tray") == (PLACE, NONE)
    assert parse_command("done") is None


def test_the_demonstration_cache_follows_the_settings_and_the_seed():
    from dataclasses import replace
    from robot_stack_earlier.armstack.cli import demo_key, demo_seed0
    from robot_stack_earlier.armstack.config import Config
    cfg = Config()
    assert demo_key(cfg) == demo_key(Config())
    assert demo_key(replace(cfg, chunk=cfg.chunk + 1)) != demo_key(cfg)        # labels depend on it
    assert demo_key(replace(cfg, seed=1)) != demo_key(cfg)
    assert demo_key(replace(cfg, wam_steps=1)) == demo_key(cfg)                 # training only
    assert demo_seed0(cfg) == 0 and demo_seed0(replace(cfg, seed=1)) >= 5_000_000 + 10_000


def test_the_held_out_split_always_leaves_something_to_train_on():
    from robot_stack_earlier.armstack.config import Config
    from robot_stack_earlier.armstack.data import Dataset
    ep = dict(success=True, sg=np.zeros((3, 2), dtype=np.int64), prop=np.zeros((4, 6), dtype=np.float32),
              tgt=np.full((3, 2), [0.36, 0.0], dtype=np.float32), holding=np.zeros(3, dtype=bool),
              frames=np.zeros((4, 96, 96, 3), dtype=np.uint8))
    ds = Dataset(Config(), [ep, ep])
    assert len(ds.train_idx) and len(ds.held_idx)
    with pytest.raises(ValueError, match="succeeded"):
        Dataset(Config(), [ep, dict(ep, success=False)])


def test_a_colour_that_is_not_on_the_table_is_an_error():
    env = ArmEnv(Config(), AnalyticController(Config()), seed=0)
    absent = next(c for c in range(10) if c not in set(env.color[:env.n].tolist()))
    with pytest.raises(ValueError):
        env.block_of_color(absent)


def test_the_surprise_count_is_of_consecutive_misses():
    from robot_stack_earlier.armstack.runtime import count_missing
    holding, missing = count_missing(False, 0, touch=1.0, expects_touch=True)
    assert (holding, missing) == (True, 0)
    holding, missing = count_missing(holding, missing, touch=0.0, expects_touch=True)
    assert missing == 1
    assert count_missing(True, 1, touch=0.0, expects_touch=True) == (True, 2)
    holding, missing = count_missing(holding, missing, touch=0.0, expects_touch=False)   # a planned release
    assert (holding, missing) == (False, 0)
    assert count_missing(holding, 0, touch=0.0, expects_touch=True) == (False, 0)      # nothing held: not a drop


def test_a_light_grip_counts_as_felt():
    """The runtime "felt" threshold is the one the demonstrations' "holding" label used."""
    from robot_stack_earlier.armstack.runtime import count_missing
    from robot_stack_earlier.armstack.sim import HELD_TOUCH
    light = (HELD_TOUCH + 0.5) / 2                     # held in the demonstrations, but under the old 0.5
    assert count_missing(True, 1, touch=light, expects_touch=True) == (True, 0)


def test_the_demonstration_cache_is_private(tmp_path, monkeypatch):
    from robot_stack_earlier.armstack.cli import demo_cache
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = demo_cache()
    assert root.parent == tmp_path and root.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("flags", [["--seed", "1"], ["--quick"], ["--steps", "5"]])
def test_eval_refuses_flags_it_would_ignore(flags, tmp_path):
    from robot_stack_earlier.armstack.cli import main
    with pytest.raises(SystemExit):
        main(["eval", "--run", str(tmp_path), *flags])


def test_train_refuses_zero_steps(tmp_path):
    from robot_stack_earlier.armstack.cli import main
    with pytest.raises(SystemExit):
        main(["train", "--run", str(tmp_path), "--steps", "0"])


def test_one_worker_collection_leaves_the_thread_count_alone():
    import torch
    cfg = replace(quick(Config()), n_demos=1)
    before = torch.get_num_threads()
    collect(cfg, train_controller(cfg, log=lambda *_: None), log=lambda *_: None)
    assert torch.get_num_threads() == before


def test_a_saved_run_loads_without_unpickling_code(tmp_path):
    from dataclasses import asdict
    import torch
    from robot_stack_earlier.armstack.cli import load
    from robot_stack_earlier.armstack.wam import WAM
    cfg = quick(Config())
    torch.save({"config": asdict(cfg), "controller": {"w": torch.zeros(2)}, "wam": WAM(cfg).state_dict(),
                "held_out": {"waypoint_mm": 1.0}}, tmp_path / "model.pt")
    loaded, ctrl, _ = load(tmp_path)
    assert loaded == cfg and torch.equal(ctrl["w"], torch.zeros(2))


@pytest.mark.parametrize("argv", [["train", "--episodes", "5"], ["eval", "--episodes", "0"]])
def test_episode_counts_are_checked(argv, tmp_path):
    from robot_stack_earlier.armstack.cli import main
    with pytest.raises(SystemExit):
        main([*argv, "--run", str(tmp_path)])


def test_say_refuses_episode_choices(tmp_path):
    from robot_stack_earlier.armstack.play import main
    with pytest.raises(SystemExit) as refused:
        main(["say", "done", "--run", str(tmp_path), "--seed", "9"])
    assert refused.value.code == 2                      # refused as a bad option, not for want of an episode
