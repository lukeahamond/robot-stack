"""Checks for the SO-101 stack that need no arm, no network and no API key.

They use the tiny settings, so they show that the pieces connect, not that the arm works.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101 import train as T
from dataclasses import asdict, replace

from so101.config import Config, from_saved, quick
from so101.controller import GRIP, Controller, Plan, TouchSensor, touch_series
from so101.data import Examples, Store, StoreWriter
from so101.planner import DONE, ClaudePlanner, ListPlanner, PlannerUnavailable
from so101.run import Delayed, Policy, Stack
from so101.sim import SeeingPlanner, SimArm, record


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("so101")
    cfg = quick()
    writer = StoreWriter(root / "store", SimArm.cameras, SimArm.motors, SimArm.fps, cfg.img)
    kept, _ = record(writer, 4, img=cfg.img, tail=cfg.tail, seed0=0, log=lambda *_: None)
    writer.close()
    ex = Examples(cfg, Store(root / "store"))
    model = T.train(cfg, ex, log=lambda *_: None, device="cpu")
    T.save(root / "model.pt", cfg, model, ex)
    return root, cfg, ex, kept


def test_touch_is_read_from_the_gripper():
    # Commanded shut (0) but stuck at 40: something is held. Commanded shut and reaching 0: nothing is.
    held = touch_series(np.array([100, 80, 60, 40, 40, 40, 40, 40.0]), np.zeros(8), lag=3, full=8.0)
    empty = touch_series(np.array([100, 80, 60, 40, 20, 0, 0, 0.0]), np.zeros(8), lag=3, full=8.0)
    assert held[-1] == 1.0 and empty[-1] == 0.0
    # While the gripper is still following an opening command it is not touch.
    opening = touch_series(np.array([0, 20, 40, 60.0]), np.array([100, 100, 100, 100.0]), lag=3, full=8.0)
    assert opening.max() == 0.0


def test_controller_limits_speed_range_and_squeeze():
    cfg = Config()
    low, high = np.array([-50.0] * 5 + [0.0]), np.array([50.0] * 5 + [100.0])
    ctl = Controller(cfg, 30, low, high)
    state = np.array([0.0, 0, 0, 0, 0, 60])
    ctl.reset(state)
    far = np.tile(np.array([500.0, 0, 0, 0, 0, 0]), (10, 1))
    ctl.set_plan(Plan(0, far, np.ones(10), -np.ones(10), cfg.ticks_per_frame), 0)
    step = cfg.max_speed / (30 * cfg.ticks_per_frame)
    prev = 0.0
    for tick in range(200):
        out = ctl.tick(tick, state)
        assert out[0] - prev <= step + 1e-9            # never faster than the limit
        assert out[GRIP] >= state[GRIP] - cfg.grip_squeeze - 1e-9   # never squeezes past the limit
        prev = out[0]
        state[0] = out[0]                              # the joint follows
    assert out[0] == pytest.approx(high[0] + cfg.limit_margin)      # and stops at the edge of the range
    ctl.hold()
    assert np.allclose(ctl.tick(201, state), out)


def test_controller_does_not_run_away_from_a_blocked_joint():
    cfg = Config()
    ctl = Controller(cfg, 30, np.array([-90.0] * 5 + [0.0]), np.array([90.0] * 5 + [100.0]))
    state = np.array([0.0, 0, 0, 0, 0, 60])            # the first joint is stuck at 0
    ctl.reset(state)
    far = np.tile(np.array([80.0, 0, 0, 0, 0, 60]), (10, 1))
    ctl.set_plan(Plan(0, far, np.ones(10), -np.ones(10), cfg.ticks_per_frame), 0)
    step = cfg.max_speed / (30 * cfg.ticks_per_frame)
    outs = [ctl.tick(tick, state)[0] for tick in range(200)]
    assert max(outs) == pytest.approx(cfg.max_lead)                # never aimed further than max_lead
    assert np.abs(np.diff(outs)).max() <= step + 1e-9              # and the target moves at the speed limit, both ways
    assert outs[-1] == pytest.approx(cfg.idle_lead)                # once it counts as blocked, only idle_lead ahead
    # Let go: the joint reaches the target, then follows the plan again with the full lead.
    state[0] = outs[-1]
    for tick in range(200, 260):
        out = ctl.tick(tick, state)
        state[0] = out[0]
    assert out[0] == pytest.approx(80.0)


def test_a_joint_that_follows_slowly_is_not_blocked():
    cfg = Config()
    ctl = Controller(cfg, 30, np.array([-90.0] * 5 + [0.0]), np.array([90.0] * 5 + [100.0]))
    state = np.array([0.0, 0, 0, 0, 0, 60])
    ctl.reset(state)
    far = np.tile(np.array([80.0, 0, 0, 0, 0, 60]), (10, 1))
    ctl.set_plan(Plan(0, far, np.ones(10), -np.ones(10), cfg.ticks_per_frame), 0)
    for tick in range(30):                              # the target is still on its way to 80 at the end
        out = ctl.tick(tick, state)
        state[0] = max(0.0, out[0] - 10.0)              # lags its target by 10 units, but keeps moving
    assert not ctl.blocked.any() and out[0] - state[0] > cfg.idle_lead


def test_a_gripper_that_starts_outside_its_range_comes_in_at_the_speed_limit():
    cfg = Config()
    ctl = Controller(cfg, 30, np.array([-50.0] * 5 + [20.0]), np.array([50.0] * 5 + [100.0]))
    state = np.array([0.0, 0, 0, 0, 0, 0.0])           # 20 below the lowest gripper value recorded
    ctl.reset(state)
    step = cfg.max_speed / (30 * cfg.ticks_per_frame)
    prev = state[GRIP]
    for tick in range(20):                              # no plan yet: the controller holds where the arm is
        out = ctl.tick(tick, state)
        assert out[GRIP] - prev <= step + 1e-9
        prev = state[GRIP] = out[GRIP]
    assert out[GRIP] == pytest.approx(20.0)
    # Inside the range the gripper is not speed limited: a plan that opens it is followed at once.
    ctl.set_plan(Plan(20, np.tile([0.0, 0, 0, 0, 0, 100], (10, 1)), -np.ones(10), -np.ones(10), cfg.ticks_per_frame), 20)
    assert ctl.tick(20, state)[GRIP] == pytest.approx(100.0)


def test_controller_target_does_not_jump_when_the_arm_is_pushed():
    cfg = Config()
    ctl = Controller(cfg, 30, np.array([-90.0] * 5 + [0.0]), np.array([90.0] * 5 + [100.0]))
    state = np.array([0.0, 0, 0, 0, 0, 60])
    ctl.reset(state)
    ctl.hold()
    step = cfg.max_speed / (30 * cfg.ticks_per_frame)
    prev = ctl.tick(0, state)
    state[0] = 80.0                                    # pushed far by hand in one tick
    for tick in range(1, 100):
        out = ctl.tick(tick, state)
        assert abs(out[0] - prev[0]) <= step + 1e-9    # the target follows at the speed limit, never at once
        prev = out
    # Held there, the joint counts as blocked, so letting go moves it back no more than idle_lead.
    assert out[0] == pytest.approx(80.0 - cfg.idle_lead)


def test_grip_and_touch_survive_the_handover_between_commands():
    """A block held at the end of one command is still squeezed and still felt when the next begins."""
    cfg = Config()
    ctl = Controller(cfg, 30, np.array([-50.0] * 5 + [0.0]), np.array([50.0] * 5 + [100.0]))
    sensor = TouchSensor(cfg)
    state = np.array([0.0, 0, 0, 0, 0, 40])            # the gripper has stopped on a block at 40
    ctl.reset(np.array([0.0, 0, 0, 0, 0, 100]))
    shut = np.zeros((30, 6))
    ctl.set_plan(Plan(0, shut, np.ones(30), -np.ones(30), cfg.ticks_per_frame), 0)
    for tick in range(40):
        squeezing = ctl.tick(tick, state)[GRIP]
        sensor.sent_command(squeezing)
    assert squeezing < state[GRIP] and sensor.read(state[GRIP]) == 1.0
    ctl.hold()
    ctl.reset(state)                                   # the next command begins, with no chunk yet
    for tick in range(40, 60):
        assert sensor.read(state[GRIP]) == 1.0
        out = ctl.tick(tick, state)
        sensor.sent_command(out[GRIP])
        assert out[GRIP] == squeezing


class ScriptedArm:
    """An arm that never moves, with a gripper reading that follows a script. For the alarm."""
    fps, cameras, motors = 25, SimArm.cameras, SimArm.motors

    def __init__(self, img, grip, sends=True):
        self.img, self.grip, self.sends, self.t, self.late_ticks = img, grip, sends, 0, 0

    def read_state(self):
        return np.array([0.0, 0, 0, 0, 0, self.grip(self.t)])

    def read_images(self):
        return {c: np.zeros((self.img, self.img, 3), dtype=np.uint8) for c in self.cameras}

    def send(self, target):
        return self.sends

    def wait_tick(self):
        self.t += 1


class FixedPolicy:
    """Always asks for a shut gripper and expects to feel something."""

    def __init__(self, cfg, norm):
        self.cfg, self.norm = cfg, norm

    def infer(self, images, state, touch, command):
        n = self.cfg.chunk
        return np.tile([0.0, 0, 0, 0, 0, 0], (n, 1)), np.ones(n), -np.ones(n)


def test_a_drop_before_the_first_chunk_of_the_next_command_raises_the_alarm(trained):
    _, cfg, ex, _ = trained
    arm = ScriptedArm(cfg.img, lambda t: 40.0)                     # shut on a block the whole time
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0)
    assert stack.run_command("carry") == "timeout" and stack.holding
    arm.grip = lambda t: 0.0                                        # dropped while the planner was thinking
    stack.runner = Delayed(10 ** 6)                                 # and the model has not answered yet
    assert stack.run_command("carry") == "surprise" and not stack.holding


def test_dry_run_commands_do_not_count_as_touch(trained):
    _, cfg, ex, _ = trained
    arm = ScriptedArm(cfg.img, lambda t: 40.0, sends=False)        # gripper resting at 40, nothing is sent
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0)
    stack.run_command("carry")
    assert not stack.holding and len(stack.sensor.sent) == 0


def test_data_training_and_saving(trained):
    root, cfg, ex, kept = trained
    assert kept >= 8
    images, prop, commands, future, chunk = ex.batch(ex.train_idx[:5], "cpu")
    assert images.shape == (5, 2, 3, cfg.img, cfg.img) and prop.shape == (5, 7)
    assert future.shape == (5, len(cfg.future), 2, 3, cfg.img, cfg.img) and chunk.shape == (5, cfg.chunk, 8)
    last = ex.store.end[0] - 1                       # the last frame of an episode is labelled finished
    assert ex.batch(np.array([last]), "cpu")[4][0, 0, 7] == 1.0
    assert ex.batch(np.array([ex.store.start[0]]), "cpu")[4][0, 0, 7] == -1.0
    cfg2, model, info = T.load(root / "model.pt")
    assert info["commands"] == ex.store.commands and info["fps"] == 25
    report = T.held_out_check(cfg2, model, ex, log=lambda *_: None)
    assert np.isfinite(list(report.values())).all()


def test_three_layers_run_and_events_reach_the_planner(trained):
    root, _, _, _ = trained
    cfg, model, info = T.load(root / "model.pt")
    cfg = replace(cfg, command_timeout=1.0, max_commands=3)
    arm = SimArm(cfg.img).reset(3)
    stack = Stack(cfg, Policy(cfg, model, info), arm, runner=Delayed(3))
    history = stack.run(arm.instruction, SeeingPlanner(arm))
    assert len(history) == 4 and all(outcome in ("finished", "surprise", "timeout") for _, outcome in history[:3])
    assert history[3][0] == "stop" and "max_commands" in history[3][1]      # the limit is in the history, not only the log
    assert len(stack.seen) == len(history)
    assert stack.stats["model_calls"] > 0
    low, high = stack.controller.low, stack.controller.high
    assert ((stack.controller.last >= low) & (stack.controller.last <= high)).all()


def test_same_movement_with_future_prediction_off(trained):
    """Leaving the future tokens out must give the same movement as hiding them."""
    import torch
    root, _, ex, _ = trained
    cfg, model, _ = T.load(root / "model.pt")
    images, prop, commands, _, _ = ex.batch(ex.held_idx[:3], "cpu")
    ctx = model.context(images, prop, model.command_ids(commands))
    act = torch.randn(3, cfg.chunk, 8)
    fut = torch.randn(3, model.n_fut, model.fut_mean.shape[0])
    t = torch.full((3,), 0.3)
    _, hidden = model.denoise(ctx, fut, act, t, torch.zeros(3, dtype=torch.bool))
    _, left_out = model.denoise(ctx, None, act, t)
    assert torch.allclose(hidden, left_out, atol=1e-4)


class FakeClient:
    """Stands in for the Anthropic client and records what was asked."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        return self.reply


def test_claude_planner_request_and_reply():
    commands = ["pick up the red block", "put it on the tray"]
    reply = SimpleNamespace(stop_reason="end_turn", content=[
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=json.dumps({"observation": "red block on the table", "command": commands[0]}))])
    client = FakeClient(reply)
    planner = ClaudePlanner(Config(), client=client)
    images = {"front": np.zeros((480, 640, 3), dtype=np.uint8), "wrist": np.zeros((480, 640, 3), dtype=np.uint8)}
    out = planner.next("Put the red block on the tray.", commands, [(commands[0], "surprise")], images)
    assert out == commands[0] and planner.last_observation == "red block on the table"
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5" and request["extra_body"] == {"fallbacks": "default"}
    schema = request["output_config"]["format"]["schema"]
    assert schema["properties"]["command"]["enum"] == commands + ["done", "stop"]
    content = request["messages"][0]["content"]
    assert [b["type"] for b in content] == ["text", "image", "text", "image", "text"]
    assert "ended with: surprise" in content[-1]["text"]
    assert Config().planner_timeout - 5 < request["timeout"] <= Config().planner_timeout
    client.reply = SimpleNamespace(stop_reason="refusal", content=[])
    with pytest.raises(PlannerUnavailable):
        planner.next("x", commands, [], images)


@pytest.mark.parametrize("content", [
    [],                                                                          # no text at all
    [SimpleNamespace(type="text", text='{"observation": "cut o')],              # not JSON
    [SimpleNamespace(type="text", text='{"observation": "x"}')],                # no command
    [SimpleNamespace(type="text", text='{"observation": "x", "command": "juggle"}')],   # not a command the arm has
])
def test_a_planner_reply_that_cannot_be_used_holds_the_arm(content):
    planner = ClaudePlanner(Config(), client=FakeClient(SimpleNamespace(stop_reason="end_turn", content=content)))
    with pytest.raises(PlannerUnavailable):
        planner.next("x", ["pick up the red block"], [], {})


def test_a_failed_planner_request_holds_the_arm():
    class Failing(FakeClient):
        def create(self, **request):
            raise TimeoutError("no answer")
    with pytest.raises(PlannerUnavailable):
        ClaudePlanner(Config(), client=Failing(None)).next("x", ["pick up the red block"], [], {})


def test_held_out_episodes_come_from_every_command(trained):
    _, _, ex, _ = trained
    s = ex.store
    held = {s.tasks[e] for e in set(s.episode[ex.held_idx].tolist())}
    several = {t for t in s.commands if s.tasks.count(t) > 1}
    assert several <= held
    for t in s.commands:                               # and every command still has something to train on
        assert any(s.tasks[e] == t for e in set(s.episode[ex.train_idx].tolist()))


def test_continuing_from_a_model_that_does_not_fit_is_refused(trained):
    import torch
    root, cfg, ex, _ = trained
    ck = torch.load(root / "model.pt", weights_only=True)
    del ck["state"]["act_out.weight"]
    torch.save(ck, root / "broken.pt")
    with pytest.raises(ValueError):
        T.train(cfg, ex, log=lambda *_: None, device="cpu", init=root / "broken.pt")


def test_list_planner_only_moves_on_after_finished():
    planner = ListPlanner(["a", "b"])
    assert planner.next("", [], [], {}) == "a"
    assert planner.next("", [], [("a", "surprise")], {}) == "a"
    assert planner.next("", [], [("a", "surprise"), ("a", "finished")], {}) == "b"
    assert planner.next("", [], [("a", "finished"), ("b", "finished")], {}) == DONE


def test_run_time_settings_come_from_now_and_trained_ones_from_the_saved_model(trained):
    root, cfg, _, _ = trained
    now = replace(Config(), max_speed=40.0, ticks_per_frame=1, planner_model="another-model")
    got, _, _ = T.load(root / "model.pt", settings=now)
    assert (got.max_speed, got.ticks_per_frame, got.planner_model) == (40.0, 1, "another-model")
    assert (got.img, got.chunk, got.width, got.encoder, got.future) == (cfg.img, cfg.chunk, cfg.width, cfg.encoder, cfg.future)
    assert T.load(root / "model.pt")[0] == cfg                       # without settings, as it was trained
    old = dict(asdict(cfg), a_setting_since_removed=1)                # a file saved by another version
    assert from_saved(old) == cfg


def test_a_command_the_model_was_not_trained_on_is_not_run(trained):
    root, _, _, _ = trained
    cfg, model, info = T.load(root / "model.pt")
    arm = SimArm(cfg.img).reset(3)
    stack = Stack(cfg, Policy(cfg, model, info), arm, runner=Delayed(3))
    history = stack.run(arm.instruction, ListPlanner(["pick up the purple blokc"]))
    assert len(history) == 1 and history[0][0] == "stop" and stack.stats["ticks"] == 0


def test_the_planner_time_limit_covers_retries():
    class APIConnectionError(Exception):
        pass

    class Down(FakeClient):
        def create(self, **request):
            self.requests.append(request)
            raise APIConnectionError("unreachable")
    client = Down(None)
    planner = ClaudePlanner(replace(Config(), planner_timeout=5.0), client=client)
    waited = []
    planner.sleep = waited.append
    with pytest.raises(PlannerUnavailable):
        planner.next("x", ["pick up the red block"], [], {})
    assert len(client.requests) == 3 and all(r["timeout"] <= 5.0 for r in client.requests) and sum(waited) < 5.0
    planner = ClaudePlanner(replace(Config(), planner_timeout=0.5), client=Down(None))       # no time to wait and try again
    with pytest.raises(PlannerUnavailable):
        planner.next("x", ["pick up the red block"], [], {})
    assert len(planner.client.requests) == 1


def test_the_planner_time_limit_is_kept_by_the_wall_clock():
    """A request that goes on past the limit is given up on, whatever the SDK's own timeout does."""
    import threading
    import time
    release = threading.Event()

    class Trickling(FakeClient):
        def create(self, **request):
            release.wait(5)
    planner = ClaudePlanner(replace(Config(), planner_timeout=0.3), client=Trickling(None))
    t0 = time.monotonic()
    try:
        with pytest.raises(PlannerUnavailable, match="too long"):
            planner.next("x", ["pick up the red block"], [], {})
    finally:
        release.set()
    assert time.monotonic() - t0 < 2.0


def test_settings_cannot_be_changed_past_the_check():
    import dataclasses
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_speed = 1e9
    with pytest.raises(ValueError, match="max_speed"):
        replace(cfg, max_speed=1e9)
    with pytest.raises(ValueError, match="warmup"):
        replace(cfg, warmup=0)


def test_continuing_on_data_with_other_words_is_refused(trained):
    root, cfg, ex, _ = trained
    other = Examples(cfg, ex.store)
    other.vocab = list(reversed(ex.vocab))                             # same number of words, another order
    with pytest.raises(ValueError, match="words"):
        T.train(cfg, other, log=lambda *_: None, device="cpu", init=root / "model.pt")


def test_the_planner_request_goes_through_the_real_sdk():
    """The stand-in client above accepts any argument. This sends the request through the installed
    SDK, to a fake HTTP server, so an argument the SDK does not take fails here."""
    anthropic = pytest.importorskip("anthropic")
    if anthropic.__version__.startswith("0."):    # anthropic 0.x is built on httpx, 1.x and later on httpx2
        import httpx
    else:
        import httpx2 as httpx
    sent = []
    commands = ["pick up the red block"]

    def handle(request):
        sent.append(dict(json.loads(request.content), beta=request.headers.get("anthropic-beta", "")))
        text = json.dumps({"observation": "x", "command": commands[0]})
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "declined part"},
                        {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-5"}},
                        {"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})
    client = anthropic.Anthropic(api_key="test", max_retries=0, base_url="http://test.invalid",
                                 http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    images = {"front": np.zeros((48, 64, 3), dtype=np.uint8)}
    for fallbacks in (True, False):
        planner = ClaudePlanner(replace(Config(), planner_fallbacks=fallbacks), client=client)
        assert planner.next("x", commands, [], images) == commands[0]          # the last text block is the answer
        assert sent[-1]["output_config"]["format"]["type"] == "json_schema"
        assert sent[-1].get("fallbacks") == ("default" if fallbacks else None)
        assert ("server-side-fallback-2026-07-01" in sent[-1]["beta"]) == fallbacks


def test_a_failing_model_call_holds_the_arm_and_ends_the_run(trained):
    from so101.run import MODEL_ERROR, Threaded
    root, _, _, _ = trained
    cfg, model, info = T.load(root / "model.pt")

    class Broken(Policy):
        def infer(self, *a):
            raise RuntimeError("out of GPU memory")
    arm = SimArm(cfg.img).reset(3)
    logged = []
    stack = Stack(cfg, Broken(cfg, model, info), arm, runner=Threaded(), log=logged.append)
    history = stack.run(arm.instruction, ListPlanner([info["commands"][0], info["commands"][0]]))
    assert history == [(info["commands"][0], MODEL_ERROR)]
    assert stack.stats["model_errors"] == 1 and stack.stats["model_calls"] == 0
    assert any("out of GPU memory" in line for line in logged)


def test_a_late_model_call_cannot_answer_the_next_one():
    import time
    from so101.run import Threaded
    runner = Threaded()
    runner.submit(lambda: (time.sleep(0.2), "old")[1], 0)
    runner.submit(lambda: "new", 1)
    while not runner.ready(1):
        time.sleep(0.001)
    time.sleep(0.3)                                    # the old call has finished by now
    assert runner.value == "new"


def test_continuing_training_keeps_the_future_scale(trained):
    import torch
    root, cfg, ex, _ = trained
    ck = torch.load(root / "model.pt", weights_only=True)
    ck["state"]["fut_mean"] += 3.0                     # a scale this data would not give, as other data would
    ck["state"]["fut_std"] *= 2.0
    torch.save(ck, root / "other_scale.pt")
    model = T.train(replace(cfg, steps=2), ex, log=lambda *_: None, device="cpu", init=root / "other_scale.pt")
    assert torch.equal(model.fut_mean, ck["state"]["fut_mean"]) and torch.equal(model.fut_std, ck["state"]["fut_std"])


def test_cameras_are_given_by_index_or_path():
    from so101.cli import camera_source
    assert camera_source("2") == 2 and camera_source(" 0 ") == 0
    assert camera_source("/dev/video2") == "/dev/video2"


def test_camera_list_ignores_spaces_and_rejects_bad_pairs():
    from so101.cli import parse_cameras
    assert parse_cameras("front=0, wrist=/dev/video2") == {"front": "0", "wrist": "/dev/video2"}
    with pytest.raises(SystemExit):
        parse_cameras("front=0,wrist")


def test_a_failing_model_call_in_simulation_ends_the_run_too(trained):
    from so101.run import MODEL_ERROR
    root, _, _, _ = trained
    cfg, model, info = T.load(root / "model.pt")

    class Broken(Policy):
        def infer(self, *a):
            raise RuntimeError("broken")
    arm = SimArm(cfg.img).reset(3)
    stack = Stack(cfg, Broken(cfg, model, info), arm, runner=Delayed(3))
    assert stack.run(arm.instruction, ListPlanner([info["commands"][0]])) == [(info["commands"][0], MODEL_ERROR)]


def test_a_model_call_that_never_answers_ends_the_run(trained, monkeypatch):
    import threading
    from so101 import run as R
    root, _, _, _ = trained
    cfg, model, info = T.load(root / "model.pt")
    cfg = replace(cfg, command_timeout=0.1)
    release = threading.Event()

    class Stuck(Policy):
        def infer(self, *a):
            release.wait(5)
            raise RuntimeError("never answered")
    monkeypatch.setattr(R, "DRAIN_SECONDS", 0.2)
    arm = SimArm(cfg.img).reset(3)
    stack = Stack(cfg, Stuck(cfg, model, info), arm, runner=R.Threaded())
    try:
        history = stack.run(arm.instruction, ListPlanner([info["commands"][0]] * 3))
    finally:
        release.set()
    assert history == [(info["commands"][0], R.MODEL_ERROR)]


def test_a_gripper_closing_on_nothing_is_not_holding(trained):
    """A gripper closing slower than the squeeze limit lags its command and reads as touch. That is not a grasp."""
    _, cfg, ex, _ = trained
    for hold_still, expect in ((quick().hold_still, False), (1e9, True)):   # 1e9: the check switched off
        arm = ScriptedArm(cfg.img, lambda t: max(0.0, 100.0 - 1.0 * t))     # closes at 50 units a second
        stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
        stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0, hold_still=hold_still)
        assert stack.run_command("pick") == "timeout"
        assert stack.holding == expect


def test_continuing_training_keeps_the_saved_scaling(trained):
    root, cfg, ex, _ = trained
    other = Examples(cfg, ex.store)
    other.norm.delta_std = other.norm.delta_std * 2                    # as data with bigger moves would give
    other.norm.low = other.norm.low - 7.0                              # and a wider range
    T.train(replace(cfg, steps=2), other, log=lambda *_: None, device="cpu", init=root / "model.pt")
    assert np.allclose(other.norm.delta_std, ex.norm.delta_std)
    assert np.allclose(other.norm.low, ex.norm.low - 7.0)


def test_commands_made_of_the_same_words_are_refused(tmp_path):
    cfg = quick()
    writer = StoreWriter(tmp_path, ["front"], SimArm.motors, 25, cfg.img)
    frames = {"front": np.zeros((20, cfg.img, cfg.img, 3), dtype=np.uint8)}
    for task in ("put red on blue", "put blue on red", "put red on blue"):
        writer.add_episode(task, np.zeros((20, 6)), np.zeros((20, 6)), frames)
    writer.close()
    with pytest.raises(ValueError, match="same words"):
        Examples(cfg, Store(tmp_path))


def test_scaling_comes_from_the_training_episodes_only(trained):
    _, cfg, ex, _ = trained
    tr = ex.train_idx
    assert np.allclose(ex.norm.state_mean, ex.store.state[tr].mean(0))
    assert len(ex.held_idx) and not np.allclose(ex.norm.state_mean, ex.store.state.mean(0))


def test_prepare_needs_a_dataset(tmp_path):
    from so101.lerobot_io import prepare
    with pytest.raises(ValueError, match="no datasets"):
        prepare([], tmp_path, 56)


def test_a_command_with_one_episode_is_never_held_out(tmp_path):
    cfg = quick()
    writer = StoreWriter(tmp_path, ["front"], SimArm.motors, 25, cfg.img)
    frames = {"front": np.zeros((20, cfg.img, cfg.img, 3), dtype=np.uint8)}
    for task in ("pick up the block", "put it on the tray"):
        writer.add_episode(task, np.zeros((20, 6)), np.zeros((20, 6)), frames)
    writer.close()
    with pytest.raises(ValueError, match="two or more episodes"):
        Examples(cfg, Store(tmp_path))


def test_a_new_chunk_mid_blend_takes_over_from_what_was_being_sent():
    cfg = Config()
    for name in ("max_speed", "max_lead"):                          # no limits, to see the blend alone. Set past the
        object.__setattr__(cfg, name, 1e9)                          # check on purpose, which a frozen Config otherwise refuses
    ctl = Controller(cfg, 30, np.full(6, -1e3), np.full(6, 1e3))
    state = np.zeros(6)
    ctl.reset(state)
    flat = lambda v: Plan(0, np.full((30, 6), v), np.ones(30), -np.ones(30), cfg.ticks_per_frame)
    ctl.set_plan(flat(0.0), 0)
    for tick in range(20):
        ctl.tick(tick, state)
    ctl.set_plan(flat(100.0), 20)                                  # blends in over 8 ticks
    for tick in range(20, 23):
        before = ctl.tick(tick, state)
    ctl.set_plan(flat(-100.0), 23)                                 # arrives while that blend is under way
    after = ctl.tick(23, state)
    assert abs(after[0] - before[0]) < 15        # 6 now; starting again from the 100 plan's raw target jumped 37


def test_waiting_for_the_planner_does_not_count_as_running_behind():
    class Clocked(ScriptedArm):
        restarts = 0

        def restart_clock(self):
            self.restarts += 1
    cfg = replace(quick(), command_timeout=0.2)
    arm = Clocked(cfg.img, lambda t: 50.0)
    norm = SimpleNamespace(low=np.full(6, -100.0), high=np.full(6, 100.0))
    stack = Stack(cfg, FixedPolicy(cfg, norm), arm, runner=Delayed(3))
    stack.run_command("carry")
    stack.run_command("carry")
    assert arm.restarts == 2


def test_a_noisy_gripper_reading_still_counts_as_holding(trained):
    """A squeezing gripper's reading jitters by a step or so. Over one tick that looks fast; it is not moving."""
    _, cfg, ex, _ = trained
    arm = ScriptedArm(cfg.img, lambda t: 40.0 + (t % 2))                # 1 unit of jitter every tick
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0)
    assert stack.run_command("pick") == "timeout"
    assert stack.holding


class EarlyFinish(FixedPolicy):
    """Says "finished" only on the first frames of each chunk."""

    def infer(self, images, state, touch, command):
        targets, touch, _ = super().infer(images, state, touch, command)
        finished = -np.ones(self.cfg.chunk)
        finished[:5] = 1.0
        return targets, touch, finished


def test_the_finished_vote_reads_only_frames_still_to_come(trained):
    """With the answer 3 frames late, frames 0-2 are past; of the next 8, only 2 say finished."""
    _, cfg, ex, _ = trained
    arm = ScriptedArm(cfg.img, lambda t: 100.0)
    stack = Stack(cfg, EarlyFinish(cfg, ex.norm), arm, runner=Delayed(3 * cfg.ticks_per_frame))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0)
    assert stack.run_command("pick") == "timeout"


def test_continuing_from_a_model_of_another_size_is_refused_clearly(trained):
    root, cfg, ex, _ = trained
    with pytest.raises(ValueError, match="other model settings"):
        T.train(replace(cfg, width=2 * cfg.width), ex, log=lambda *_: None, device="cpu", init=root / "model.pt")


def test_a_saved_dinov2_model_loads_without_fetching_and_gives_the_same_output(trained, tmp_path, monkeypatch):
    """The real encoder is rebuilt from the settings saved with the model. A small DINOv2 stands in for the
    pretrained one, so nothing is downloaded."""
    import torch
    transformers = pytest.importorskip("transformers")
    from so101.wam import WAM
    _, _, ex, _ = trained
    small = transformers.Dinov2Config(hidden_size=48, num_hidden_layers=2, num_attention_heads=2,
                                      intermediate_size=96, image_size=56, patch_size=14).to_diff_dict()
    cfg = replace(quick(), encoder="dinov2_small", unfreeze=1)
    torch.manual_seed(0)
    model = WAM(cfg, ex.store.cameras, ex.vocab, encoder_shape=small).eval()
    T.save(tmp_path / "dino.pt", cfg, model, ex)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        lambda *a, **k: pytest.fail("loading a saved model must not fetch the pretrained encoder"))
    _, loaded, _ = T.load(tmp_path / "dino.pt")
    images, prop, commands, _, _ = ex.batch(ex.held_idx[:2], "cpu")
    cmd = model.command_ids(commands)
    with torch.no_grad():
        assert torch.allclose(model.context(images, prop, cmd), loaded.context(images, prop, cmd), atol=1e-5)


@pytest.mark.parametrize("bad", [dict(max_speed=-50.0), dict(max_speed=0.0), dict(max_lead=-1.0),
                                 dict(ticks_per_frame=0), dict(execute=30), dict(blend=13),
                                 dict(grip_squeeze=8.0), dict(grip_squeeze=4.0, idle_squeeze=4.0),
                                 dict(idle_squeeze=30.0), dict(idle_squeeze=8.0), dict(steps=0), dict(prefetch=-1),
                                 dict(touch_lag=0), dict(hold_still=0.0), dict(denoise_steps=0), dict(tail=0),
                                 dict(planner_image=0), dict(max_speed=1e9), dict(img=100), dict(idle_lead=0.0),
                                 dict(idle_lead=40.0), dict(joint_hot=0.0), dict(shift=-1), dict(block_time=0.0)])
def test_unsafe_settings_are_refused(bad):
    with pytest.raises(ValueError, match="bad settings"):
        replace(Config(), **bad)


def test_the_shipped_settings_are_accepted():
    from so101.config import sim
    for cfg in (Config(), sim(), quick()):
        assert cfg.execute < cfg.chunk and cfg.grip_squeeze > cfg.touch_full


class RecordingArm(ScriptedArm):
    def __init__(self, *a, temperature=None, **k):
        super().__init__(*a, **k)
        self.sent = []
        if temperature is not None:
            self.temperatures = lambda: {"shoulder_lift": 30.0, "gripper": temperature()} if callable(temperature) else temperature

    def send(self, target):
        self.sent.append(np.array(target))
        return self.sends


def test_the_gripper_eases_its_squeeze_while_nothing_runs_and_still_feels_the_block(trained):
    _, cfg, ex, _ = trained
    arm = RecordingArm(cfg.img, lambda t: 40.0)                     # shut on a block
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0)
    assert stack.run_command("carry") == "timeout" and stack.holding
    assert stack.controller.last[GRIP] == 40.0 - stack.cfg.grip_squeeze
    n = len(arm.sent)
    stack.ease()
    assert len(arm.sent) == n + 1 and arm.sent[-1][GRIP] == 40.0 - stack.cfg.idle_squeeze
    assert stack.sensor.read(40.0) == 1.0                           # still full touch
    stack.ease()
    assert len(arm.sent) == n + 1                                   # already eased: nothing more is sent
    assert stack.run_command("carry") == "timeout" and stack.holding and stack.stats["surprises"] == 0


def test_a_hot_gripper_motor_ends_the_run(trained):
    _, cfg, ex, _ = trained
    commands = ex.store.commands
    for temperature, stops in ((lambda: 75.0, True), (lambda: 35.0, False)):
        arm = RecordingArm(cfg.img, lambda t: 40.0, temperature=temperature)
        stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
        stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=0.2, max_commands=2)
        history = stack.run("carry", ListPlanner([commands[0]]), commands)
        assert (history[-1][0] == "stop" and "gripper motor at 75" in history[-1][1]) == stops
        # Hot from the start: stopped after the planner's first answer, before anything moves. Else two commands
        # and the note that max_commands was reached.
        assert len(history) == (1 if stops else 3)


def test_a_failed_temperature_reading_does_not_stop_the_run(trained):
    _, cfg, ex, _ = trained

    def broken():
        raise OSError("no answer from motor 6")
    arm = RecordingArm(cfg.img, lambda t: 40.0, temperature=broken)
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3), log=(logged := []).append)
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=0.2, max_commands=1)
    history = stack.run("carry", ListPlanner([ex.store.commands[0]]), ex.store.commands)
    assert len(history) == 2 and history[0][1] == "timeout" and "max_commands" in history[1][1]
    assert any("temperature" in line for line in logged)


def test_temperatures_that_cannot_be_read_again_and_again_end_the_run(trained):
    _, cfg, ex, _ = trained
    from so101.run import TEMPERATURE_MISSES
    answers = iter([OSError("no answer"), 35.0, OSError("no answer"), OSError("no answer"), OSError("no answer")])

    def flaky():
        value = next(answers)
        if isinstance(value, Exception):
            raise value
        return value
    arm = RecordingArm(cfg.img, lambda t: 40.0, temperature=flaky)
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=0.2, max_commands=5)
    command = ex.store.commands[0]
    history = stack.run("carry", ListPlanner([command]), ex.store.commands)
    # One miss, then a good reading, starts the count again. Three misses in a row end the run: with no
    # readings the heat check is off, and the arm should not go on without it.
    assert TEMPERATURE_MISSES == 3
    assert history == [(command, "timeout"), (command, "timeout"),
                       ("stop", "the motors' temperatures could not be read 3 times in a row")]


def test_prefetching_gives_the_same_batches_in_the_same_order():
    def batches(ahead):
        rng = np.random.default_rng(3)
        return [b.tolist() for b in T.prefetched(lambda: rng.integers(0, 100, 4), 20, ahead)]
    assert batches(0) == batches(2) == batches(5)


def test_a_failure_while_prefetching_reaches_the_training_loop():
    calls = []

    def make():
        calls.append(1)
        if len(calls) == 3:
            raise OSError("disk read failed")
        return len(calls)
    got = []
    with pytest.raises(OSError, match="disk read failed"):
        for b in T.prefetched(make, 10, 2):
            got.append(b)
    assert got == [1, 2]


class NaNPolicy(FixedPolicy):
    """Answers with joint targets that are not numbers, as a model can when its 16-bit floats overflow."""

    def infer(self, images, state, touch, command):
        targets, touch, finished = super().infer(images, state, touch, command)
        targets[:, 1] = np.nan
        return targets, touch, finished


def test_a_model_answer_that_is_not_numbers_holds_the_arm_and_ends_the_run():
    from so101.run import MODEL_ERROR
    cfg = quick()
    sent = []

    class Recording(ScriptedArm):
        def send(self, target):
            sent.append(np.array(target))
            return True
    arm = Recording(cfg.img, lambda t: 50.0)
    norm = SimpleNamespace(low=np.full(6, -100.0), high=np.full(6, 100.0))
    logged = []
    stack = Stack(cfg, NaNPolicy(cfg, norm), arm, runner=Delayed(3), log=logged.append)
    history = stack.run("carry it", ListPlanner(["carry", "carry"]), commands=["carry"])
    assert history == [("carry", MODEL_ERROR)]
    assert stack.stats["model_errors"] == 1 and stack.stats["model_calls"] == 0
    assert sent and all(np.isfinite(t).all() for t in sent)        # nothing that is not a number reached the motors
    assert np.isfinite(stack.controller.last).all()                 # so easing the grip afterwards sends numbers too
    assert any("not numbers" in line for line in logged)


def test_the_controller_refuses_targets_that_are_not_numbers():
    cfg = Config()
    ctl = Controller(cfg, 30, np.full(6, -50.0), np.full(6, 50.0))
    state = np.zeros(6)
    ctl.reset(state)
    good = ctl.last.copy()
    bad = np.zeros((30, 6))
    bad[:, 2] = np.nan
    ctl.set_plan(Plan(0, bad, np.zeros(30), -np.ones(30), cfg.ticks_per_frame), 0)
    with pytest.raises(ValueError, match="not numbers"):
        ctl.tick(0, state)
    assert np.array_equal(ctl.last, good)


def test_the_timeout_is_kept_by_the_wall_clock_when_the_loop_runs_behind():
    import time

    class Slow(ScriptedArm):
        """A real-time arm (it has a clock) whose every tick takes far longer than it should."""

        def restart_clock(self):
            pass

        def wait_tick(self):
            time.sleep(0.03)
            super().wait_tick()
    cfg = replace(quick(), command_timeout=0.3)
    arm = Slow(cfg.img, lambda t: 40.0)
    norm = SimpleNamespace(low=np.full(6, -100.0), high=np.full(6, 100.0))
    stack = Stack(cfg, FixedPolicy(cfg, norm), arm, runner=Delayed(3))
    assert stack.run_command("carry") == "timeout"
    # By ticks the timeout is 0.3 s x 50 ticks a second = 15 ticks, which at 0.03 s each would be 0.45 s.
    # Ending in fewer ticks shows the wall clock ended it; the time itself is not checked, so a busy
    # machine cannot fail this.
    assert stack.stats["ticks"] < 0.3 * stack.tick_hz


class PushedArm(RecordingArm):
    """Joints that stay where they are whatever is sent, as if held against something."""


class ReachingPolicy(FixedPolicy):
    """Asks every joint to go far from where the arm is."""

    def infer(self, images, state, touch, command):
        n = self.cfg.chunk
        return np.tile([40.0, -40, 40, -40, 40, 0], (n, 1)), np.ones(n), -np.ones(n)


def test_a_blocked_joint_is_eased_while_nothing_runs(trained):
    _, cfg, ex, _ = trained
    arm = PushedArm(cfg.img, lambda t: 40.0)
    stack = Stack(cfg, ReachingPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    # block_time longer than the command, so the joints are still aimed up to max_lead when it ends.
    # (With the shipped block_time they would already be down to idle_lead: see the test below.)
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0, block_time=10.0)
    assert stack.run_command("carry") == "timeout"
    lead, idle = stack.cfg.max_lead, stack.cfg.idle_lead
    assert np.abs(stack.controller.last[:GRIP]).max() > idle        # pushing past the idle limit, up to max_lead
    assert np.abs(stack.controller.last[:GRIP]).max() <= lead
    pushing = stack.controller.last[:GRIP].copy()
    assert (np.abs(pushing) > idle).all()
    stack.ease()
    assert np.allclose(arm.sent[-1][:GRIP], np.sign(pushing) * idle)   # the arm is at 0: aimed `idle_lead` the same way
    n = len(arm.sent)
    stack.ease()
    assert len(arm.sent) == n                                       # already eased


def test_a_joint_blocked_during_a_command_is_aimed_only_idle_lead_ahead(trained):
    _, cfg, ex, _ = trained
    arm = PushedArm(cfg.img, lambda t: 40.0)
    stack = Stack(cfg, ReachingPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=1.0)
    assert stack.run_command("carry") == "timeout"
    assert np.abs(np.array(arm.sent)[:, :GRIP]).max() <= stack.cfg.max_lead + 1e-9
    assert np.allclose(np.abs(arm.sent[-1][:GRIP]), stack.cfg.idle_lead)


def test_easing_never_sends_a_reading_that_is_not_a_number(trained):
    _, cfg, ex, _ = trained
    arm = RecordingArm(cfg.img, lambda t: 40.0)
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3), log=(logged := []).append)
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=0.2)
    stack.run_command("carry")
    before, n = stack.controller.last.copy(), len(arm.sent)
    arm.read_state = lambda: np.array([np.nan, 0, 0, 0, 0, 40.0])
    stack.ease()
    assert len(arm.sent) == n and np.array_equal(stack.controller.last, before)
    assert any("could not ease" in line for line in logged)


def test_a_hot_joint_motor_ends_the_run(trained):
    _, cfg, ex, _ = trained
    arm = RecordingArm(cfg.img, lambda t: 40.0, temperature={"elbow_flex": 70.0, "gripper": 35.0})
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=0.2, max_commands=2)
    history = stack.run("carry", ListPlanner([ex.store.commands[0]]), ex.store.commands)
    assert history[-1][0] == "stop" and "elbow_flex motor at 70 C (joint_hot is 65 C)" in history[-1][1]


def test_what_the_planner_saw_is_kept_beside_the_history(trained):
    _, cfg, ex, _ = trained
    commands = ex.store.commands

    class Talking(ListPlanner):
        def next(self, *a):
            self.last_observation = f"look {len(a[2])}"
            return super().next(*a)
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), RecordingArm(cfg.img, lambda t: 40.0), runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=0.2, max_commands=2)
    history = stack.run("carry", Talking([commands[0]]), commands)
    assert stack.seen == ["look 0", "look 1", ""] and len(history) == 3      # the last entry is the max_commands note


def test_sampling_refuses_zero_denoising_steps(trained):
    root, _, ex, _ = trained
    _, model, _ = T.load(root / "model.pt")
    images, prop, commands, _, _ = ex.batch(ex.held_idx[:1], "cpu")
    with pytest.raises(ValueError, match="1 step or more"):
        model.sample(images, prop, model.command_ids(commands), steps=0)


def test_a_gripper_reading_above_its_range_is_not_squeezed_past_the_limit():
    cfg = Config()
    ctl = Controller(cfg, 30, np.array([-50.0] * 5 + [0.0]), np.array([50.0] * 5 + [60.0]))
    state = np.array([0.0, 0, 0, 0, 0, 100.0])         # the gripper reads 40 above the highest value recorded
    ctl.reset(state)
    ctl.set_plan(Plan(0, np.tile(state, (10, 1)), np.ones(10), -np.ones(10), cfg.ticks_per_frame), 0)
    for tick in range(20):
        out = ctl.tick(tick, state)
        assert out[GRIP] >= state[GRIP] - cfg.grip_squeeze - 1e-9


def test_the_drop_alarm_follows_the_chunk_being_blended_in():
    cfg = Config()
    ctl = Controller(cfg, 30, np.full(6, -1e3), np.full(6, 1e3))
    ctl.reset(np.zeros(6))
    assert ctl.expected_touch(0) is None                # no plan yet
    flat = lambda touch, tick0: Plan(tick0, np.zeros((30, 6)), np.full(30, touch), -np.ones(30), cfg.ticks_per_frame)
    ctl.set_plan(flat(1.0, 0), 0)
    assert ctl.expected_touch(5) > 0
    ctl.set_plan(flat(-1.0, 20), 20)                    # a release blends in over 8 ticks
    span = cfg.blend * cfg.ticks_per_frame
    assert ctl.expected_touch(20) > 0                   # the old chunk still sends most of the targets
    assert ctl.expected_touch(20 + span) < 0            # the new one has taken over


def test_motors_that_heat_up_while_the_planner_thinks_stop_the_run_before_moving(trained):
    _, cfg, ex, _ = trained
    arm = RecordingArm(cfg.img, lambda t: 40.0, temperature=lambda: 75.0)   # hot by the time the planner answers
    stack = Stack(cfg, FixedPolicy(cfg, ex.norm), arm, runner=Delayed(3))
    stack.cfg = stack.controller.cfg = replace(quick(), command_timeout=0.2, max_commands=2)
    history = stack.run("carry", ListPlanner([ex.store.commands[0]]), ex.store.commands)
    assert history == [("stop", "gripper motor at 75 C (grip_hot is 60 C)")] and arm.sent == []


def test_shift_zero_turns_the_image_shift_off(trained):
    import torch
    _, cfg, ex, _ = trained
    no_shift = Examples(replace(cfg, shift=0), ex.store)
    idx = ex.train_idx[:4]
    plain, *_ = no_shift.batch(idx, "cpu")
    shifted, *_ = no_shift.batch(idx, "cpu", augment=True, rng=np.random.default_rng(0))
    ratio = shifted / plain.clamp(min=1e-6)
    # Only the brightness changes: one gain per example, the same at every pixel that is not clipped.
    for r, p in zip(ratio, plain, strict=True):
        ok = (p > 0.05) & (p < 0.8)
        assert torch.allclose(r[ok], r[ok].median().expand_as(r[ok]), atol=1e-4)
