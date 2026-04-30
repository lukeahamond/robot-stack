# A three-layer robot stack on a real arm

[![tests](https://github.com/lukeahamond/robot-stack/actions/workflows/tests.yml/badge.svg)](https://github.com/lukeahamond/robot-stack/actions/workflows/tests.yml)

A robot arm that is told what to do in words and then does it. Three models run at three
speeds: a planner that looks and decides, a world action model that turns a command into
movement, and a controller that drives the motors. The stack was built for one SO-ARM101
and runs on it.

[![The SO-101 putting yellow blocks in a round tray](results/so101/result.gif)](results/so101/result.mp4)

A clip from the session at double speed. Click it for the
[full-quality video](results/so101/result.mp4), which is stored with Git LFS: a clone
without Git LFS installed gets a small pointer file in its place.

## Run it

Recording demonstrations, training and running on the arm are covered step by step in
[SO101.md](SO101.md). The tests need no arm, no network and no API key:

```bash
python3 -m pytest -q
```

```bash
ruff check .
```

Needs Python 3.10 or later, PyTorch, MuJoCo, NumPy and Pillow. The real-arm code in `so101/`
needs Python 3.12 or later and the extras listed in [SO101.md](SO101.md).

## In this repository

| Path | What it is |
|---|---|
| [SO101.md](SO101.md), `so101/` | The stack for a real SO-ARM101 and a 24 GB GPU, with the Claude API as planner. The automated tests use a simulated stand-in arm. |
| [ARMSTACK.md](robot_stack_earlier/ARMSTACK.md), `robot_stack_earlier/armstack/` | An earlier version on a simulated arm, with the lower two layers trained from scratch. The stand-in arm in `so101/sim.py` imports from it, so it is still in use. |
| [ROBOSTACK.md](robot_stack_earlier/ROBOSTACK.md), `robot_stack_earlier/robostack/` | An earlier five-part version with a critic and a rollout check. Those two parts were left out of the later design. |
| `tests/` | Tests for all three, on simulated arms. |
| [Hugging Face](https://huggingface.co/lukehamond/robot-stack) | The trained model. Private for now. |
| [LICENSE](LICENSE) | MIT. |
