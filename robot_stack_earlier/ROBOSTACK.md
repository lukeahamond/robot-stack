# robostack

An earlier simulation, kept for reference. The current work is [so101](../SO101.md). Run the
commands below from the top of the repository.

A simulation-scale scaffold of a five-part robot stack: a slow planner, a fast action
model, a controller, a critic, and a rollout check that runs only before irreversible
steps. It trains from scratch on a laptop in about ten minutes and ends with an ablation
table.

It is a scaffold, not a result. The simulator is a toy, the models are small MLPs on top
of a small convolutional encoder, and the numbers say whether the pieces fit together,
not whether the design beats alternatives on real hardware.

## Run it

```bash
python3 -m robot_stack_earlier.robostack.cli all
```

```bash
python3 -m robot_stack_earlier.robostack.cli all --quick
```

```bash
python3 -m pytest -q tests/test_smoke.py
```

The first command collects data, trains every stage, evaluates, runs RL fine-tuning and
evaluates again. `--quick` does the same with tiny settings, to check the plumbing; it still evaluates 500
episodes per variant, so it takes about a minute and a half on a laptop (pass `--episodes`
to shorten it). Needs Python 3.10+, NumPy and PyTorch. Results land in `runs/default/`.

## What is in the loop

| Part | Rate | What it does here | What it stands in for |
|---|---|---|---|
| Planner | On events | Emits the next subgoal as words (`pick red`) and as a predicted future latent | A vision-language model |
| Action model | Every 2 actions | Maps frame latent, proprioception and subgoal to a chunk of 4 actions; also predicts the future latent as a side objective | A vision-language-action model |
| Controller | 4 ticks per action | Tracks each waypoint | A joint-level controller trained in simulation |
| Critic | After each chunk | From the real frame: is the subgoal done, has a finished subgoal come undone. It also learns how likely success is, which is used only to weight the RL fine-tuning, not while running | A value and success model |
| Rollout check | Only when a chunk would release a held object | Imagines the chunk with an ensemble world model; vetoes on predicted failure or on disagreement | Imagined rollouts from a world model |

The planner runs when the critic reports the subgoal done, when a finished subgoal has
regressed (a slip), on a timeout, or after repeated vetoes. Without the critic it runs
on a fixed timer.

## The task

A gripper moves over a table with one or two coloured blocks and three coloured bins. The
instruction names a bin for each block. Three things make it a useful miniature:

- two-block tasks need sequencing, so there is something to plan;
- a carried block can slip, which is recoverable but has to be noticed;
- releasing over the wrong bin is irreversible and ends the episode.

Observations are a 32x32 camera image and four proprioceptive numbers: gripper x, y,
commanded-closed, and a contact sensor.

## Training stages

1. **Data.** A scripted expert stands in for teleoperation. Expert episodes are perturbed
   (disturbed actions, grabbing the other block) while the expert's intended action is
   kept as the label. Deliberately sloppy episodes supply failures.
2. **Encoder.** Reconstruction plus action prediction, with keypoint features. Then
   frozen: every other model reads and predicts in this latent space.
3. **World model.** An ensemble of latent dynamics models with an irreversible-failure head.
4. **Planner.** Hindsight labels: the subgoal in force, and the latent of the frame where
   that subgoal was achieved.
5. **Action model.** Behaviour cloning, with each interface channel randomly dropped so
   neither words nor latents can be ignored.
6. **Critic.** Trained on successes and failures.
7. **Correction rounds.** The agent acts, the expert relabels every state it visits,
   everything is refitted. Stands in for a person taking over when the robot goes wrong.
8. **RL fine-tuning.** The agent acts alone with exploration noise, the critic is refitted
   on the outcomes, and the action model is updated by advantage-weighted regression. No
   gradient passes through the world model.

## What building it showed

Each of these was a failure first, found by tracing episodes, and each has a direct
counterpart on real robots.

- **The planner could not tell a grasp from a miss.** At 32x32 a held block and a block
  beside a closed gripper look the same, until a contact sensor was added to proprioception.
- **Clean demonstrations teach shortcuts.** With only clean data the planner learned
  "gripper closed means the commanded block is held". Perturbed demonstrations with clean
  labels were needed to break that.
- **The frozen encoder set the ceiling on precision.** Reconstruction alone located blocks
  to within a few pixels, too coarse to grasp. Adding action prediction to the encoder's
  objective and keypoint features helped; more demonstrations helped more.
- **Imitation alone does not recover.** A small position error carried the gripper into a
  region no demonstration visited and it stayed there. Correction rounds were added for this.

## Where the toy is not the real thing

- **The expert reads simulator state.** Demonstrations, corrections and the critic's
  subgoal labels all come from privileged state. On a robot these are teleoperation,
  human interventions, and human or model-generated labels.
- **The world model is not in the data pipeline.** In the design this scaffolds, an
  offline world model generates varied practice runs. Here the simulator plays that role
  directly, and the learned world model is used only for the rollout check.
- **Words are two tokens.** A subgoal is a verb and a colour. A real planner emits free
  text and the action model needs a language encoder.
- **One embodiment, one camera, no physics.** There is no contact dynamics, occlusion or
  depth.
- **The planner is cheap here.** In the real system it is the slowest and costliest part,
  which is the reason to call it on events. The table reports planner calls so that cost
  is visible, but it is not charged against success.

## Layout

```
robostack/
  config.py      every setting in one dataclass
  env.py         the simulator
  expert.py      scripted demonstrator and the ground-truth subgoal
  controller.py  fast waypoint tracker
  data.py        episode recording, collection, the training buffer
  models.py      encoder, planner, action model, critic, world model
  train.py       the eight stages above
  runtime.py     the agent loop and the ablation harness
  cli.py         command line
tests/           simulator rules and an end-to-end pipeline check
```

## Moving to a real benchmark

- **Simulator.** Replace `ToyKitchen` with an adapter around ManiSkill or LIBERO exposing
  `reset`, `tick`, `render`, `proprio`, `instruction`, `achieved`, `done`, `success`,
  `failed`.
- **Encoder.** Replace `Encoder` with a pretrained video encoder and delete the encoder
  stage.
- **Planner and action model.** Replace with a pretrained vision-language model and an
  open vision-language-action model, fine-tuned through the same `train_planner` and
  `train_action` losses.
- **Subgoal labels.** Replace `achieved` with captions from a vision-language model.
