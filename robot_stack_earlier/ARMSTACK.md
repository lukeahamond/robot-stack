# armstack

An earlier simulation. The current work is [so101](../SO101.md). It is still in use:
`so101/sim.py` builds its simulated stand-in arm on this package, so changing it changes the
so101 stand-in too. Run the
commands below from the top of the repository.

A simulated robot arm run by three layers, with the lower two trained here from scratch:

| Layer | Rate | What it is | Trained on |
|---|---|---|---|
| Planner | On events | Whoever gives the next command in words. With `play` (below) that is you, typing commands; in `eval` it is a scripted planner. | Nothing here |
| World action model | Every 0.8 s | A 3.5-million-parameter transformer. From the camera frame, body state (including touch) and the command, it generates the camera frames 0.8 s and 1.6 s ahead (or the frame where the command ends, if that is sooner), the next eight hand positions and gripper decisions, the touch it expects, and a "command finished" flag, in one denoising pass. | 1,500 scripted demonstrations |
| Controller | 50 Hz | A 34,000-parameter network that turns "put the fingertip here" into joint targets. | Arm geometry, 6 seconds |

It is a small-scale check that the pieces fit together. It says nothing about how a
full-size system would perform.

## Run it

```bash
python3 -m robot_stack_earlier.armstack.cli train
```

```bash
python3 -m robot_stack_earlier.armstack.cli eval
```

```bash
python3 -m pytest -q tests/test_arm.py
```

Training takes about 50 minutes on an Apple M4 laptop (two minutes of demonstrations, the
rest the world action model). Needs Python 3.10+, PyTorch, MuJoCo and Pillow. Outputs land
in `runs/arm/`.

To give the commands yourself:

```bash
python3 -m robot_stack_earlier.armstack.play new --seed 5000007 --drop
```

```bash
python3 -m robot_stack_earlier.armstack.play say "pick up the red block"
```

Each call saves the camera view to `runs/arm/play/view.png` and stops at the next event.
The commands it understands are "pick up the red/green/blue block", "put it on the tray"
and "done".

## The task

A five-joint arm with a two-finger gripper, in MuJoCo. Two or three coloured blocks sit on
a table and the instruction names one or two of them to put on a tray, in order. In the
disturbed condition the gripper is forced open once while carrying, so the block falls
back onto the table.

## What went wrong on the way

The first full training run failed. Four things had to change.

- **The frame generator could not work as designed.** A generated image patch holds 432
  numbers and the model is 192 wide, so it could not carry the noise it was asked to
  predict. It now predicts the clean frames and actions directly.
- **The demonstrator was too fussy.** It only descended within 6 mm of a block and backed
  off otherwise. The camera resolves about 5 mm per pixel, so the same picture was
  labelled "go down" in one demonstration and "back up" in another. A plain image network
  could not learn from that data either. The demonstrator now approaches in a smooth
  funnel.
- **The demonstrations never showed a missed grasp.** A half-trained model closed on
  nothing, expected to feel the block, and reported success. Demonstrations now include
  deliberate misses, labelled with the correct recovery.
- **Labels were stitched from noisy futures.** Each example's eight-step plan is now what
  the demonstrator would do from exactly that state.

All four were about the data or the training target. None was about the three-layer design.

## Limits

- **The generated frames are poor.** They show the table, the tray and a blurred arm, and
  mostly leave out the blocks. The action half of the
  model learned well; the picture half did not, in 50 minutes at this size. No run here
  tests whether generating frames helped the actions at all.
- **The planner's words are reduced to two tokens.** Rephrasing a command changes nothing.
- **The demonstrator reads the simulator's true state.** On a real robot this is
  teleoperation.
- **The controller copied exact arm geometry.** It was not learned by trial and error, and
  an arm is far easier than whole-body balance.
- **One task, one arm, one camera, one training run.** Run-to-run variation was not measured.
- **"Touch" is a simulated contact force on the finger pads,** reduced to one number.

## Layout

```
armstack/
  config.py    every setting
  sim.py       the arm, blocks, tray and camera in MuJoCo
  control.py   the 50 Hz controller and its training
  expert.py    scripted demonstrator and its roll-forward plan
  data.py      recording demonstrations, building training examples
  wam.py       the world action model
  train.py     training and the held-out check
  runtime.py   the three layers running together, scripted planners, evaluation
  play.py      give the commands by hand
  film.py      save generated frames next to what the camera then showed
  cli.py       train and eval commands
tests/test_arm.py
```
