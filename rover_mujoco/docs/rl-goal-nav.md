# RL goal navigation: drive to a coordinate, avoid what's in the way

`GoalNav-v0` is the second RL task in this repo, and it is deliberately built on the same
footing as `docs/rl-line-follower.md`'s `LineFollowerReal-v0`: same rover, same action space,
same sim-to-real constraints, same domain-randomization ranges. What changes is the job.
Instead of "keep a line centered in the camera", the command is a coordinate:

> Drive to (1, 2) — 1 metre to your right, 2 metres forward.

Nothing in the world marks that spot. The rover has to get there by dead reckoning, inside a
walled 5×5 m arena with obstacles scattered around it.

## Why this task is interesting

The line follower is a *reactive* task: everything the policy needs is in the current frame,
and a policy with no memory can do it well. Goal navigation is not. The goal is a place, and
knowing where you are relative to a place means integrating your own motion over time — which
means the error integrates too. That is the whole lesson here, and it's why the env hands the
policy a goal vector that drifts rather than a correct one.

## Observation: exactly what the rover can sense

| Field | Shape | What it is |
|-------|-------|-----------|
| `image` | (64, 64, 1) uint8 | The onboard camera, through the same randomization pipeline as `LineFollowerRealEnv` |
| `encoders` | (2,) | Left/right wheel ticks accrued since the last control step |
| `lidar` | (3,) | Left/centre/right beam distances in metres, clipped to 2 m |
| `goal` | (2,) | (forward, left) metres to the goal **in the rover's estimated body frame** |

The lidar is not an invention for the sim. The physical rover already has it: the firmware's
`"l"` UDP command answers with `left_distance`, `center_distance`, `right_distance` and a
`center_ok` validity flag, which `rover_control/encoder_poller.py` polls today. Modelling it
in MuJoCo is three `<rangefinder>` sensors on three sites in `rover.xml`, splayed
`LIDAR_SPLAY_DEG` either side of straight ahead.

That poller *alternates* its encoder and lidar queries, so each sensor lands at half the poll
rate. `rl_rover.py` already polls encoders at the full rate for the deployed policy, which
leaves the lidar arriving every other control step — so the env holds the previous lidar
reading on odd steps (`LIDAR_DECIMATION`) rather than pretending both sensors are fresh.

### The camera is the weak sensor here, and that's measured

The line follower's camera sits at ~15.2° from straight down, which is right for looking at a
line under the rover and useless for looking at an obstacle in front of it. Rendering a box at
various mount angles:

| Tilt | Obstacle 0.4 m ahead | Obstacle 0.8 m ahead |
|------|---------------------|---------------------|
| 15.2° (line-follower default) | 0.0% of pixels | 0.0% |
| 45° | 6.0% | 0.0% |
| 60° (`CAMERA_ANGLE_DEG` here) | 8.4% | 1.1% |
| 75° | 9.4% | 1.0% |

So the camera is a short-range confirmation at best; the lidar is the obstacle sensor. Worth
knowing before you spend a training run wondering why vision isn't carrying the task.

## The goal vector drifts, on purpose

`envs/arena.py`'s `integrate_odometry()` dead-reckons the pose from wheel ticks, and it is fed
**the same noisy, quantized ticks the policy sees** — not ground truth. Measured drift after
~12 s of driving with domain randomization on: ~0.10 m mean, 0.13 m worst case across seeds,
against a `GOAL_TOLERANCE_M` of 0.25 m. Tight enough to be solvable, loose enough that a
policy which ignores its sensors and drives open-loop will miss.

Two sign conventions in `rover.xml` have to be undone to make that integration correct, and
both are easy to get wrong (the first version of this env had a mirrored turn direction):

1. The right wheel's body frame is mirrored, so equal-sign wheel commands spin in place.
2. The joint **names** are swapped relative to the physical sides — `left_axle` is the wheel
   at x=+0.0798, which is on the rover's *right* when forward is +Y.

`envs/arena.py` documents both at the point where they're undone. The integrator itself uses
the midpoint rule rather than "turn fully, then translate", so the drift you see comes from
the sensor rather than from a sloppy integrator.

## The arena

5×5 m, walled, with 3–8 obstacles per episode drawn from a fixed pool (MuJoCo compiles
geometry once, so `GoalNavEnv` moves and resizes pool members each reset — the same runtime
model-mutation trick `LineFollowerEnv` uses for appearance randomization). Obstacles are
rejection-sampled so they never land on the spawn, the goal, or each other. Both the spawn
pose and the goal are randomized, so there is no fixed layout to memorize.

The chassis needed a collision proxy for any of this to mean anything. Every chassis geom that
came out of the URDF import is visual-only (`contype`/`conaffinity` 0), so before
`chassis_collision` was added to `rover.xml` the only collidable parts were the three wheels,
and the rover drove straight through anything shorter than its deck. That proxy uses a
`contype`/`conaffinity` bitmask of 2, which the arena's walls and obstacles opt into and the
floor does not — so the line-follower scenes that share `rover.xml` keep exactly the contact
set they were trained against. (Verified: both scenes settle to the same height with the same
contact count.)

## Reward

Shaped like the line follower's, for the same reasons:

- **Progress** dominates: `PROGRESS_WEIGHT * (distance closed this step) / (initial distance)`,
  normalized so a complete run sums to ~`PROGRESS_WEIGHT` whether the goal was sampled 1 m or
  3 m away.
- **`SUCCESS_BONUS`** on arriving within `GOAL_TOLERANCE_M`.
- **`COLLISION_PENALTY`**, and the episode ends there.
- **`STEP_PENALTY`** every step, so dawdling costs something and standing still is never
  optimal — the failure mode that made the first line-follower reward function degenerate.

Reward uses ground-truth position. That is the same reward-only privilege `LineFollowerEnv`
documents: reward is a training-time construct that doesn't exist at deployment, unlike the
observation, which stays honest.

## Running it

```shell
cd rover_mujoco
uv run python scripts/gen_arena.py             # regenerate the arena MJCF (only after editing envs/arena.py)
uv run python scripts/train_goal_nav.py        # saves to runs/ppo_goal_nav/
uv run python scripts/evaluate_goal_nav.py     # viewer, random goals
uv run python scripts/evaluate_goal_nav.py 1 2 # viewer, "drive to (1, 2)"
```

## What to experiment with

- **Does vision help at all?** Drop `image` from the observation and see whether lidar +
  odometry alone matches it. Given the numbers above, it might — and a much smaller
  observation trains a lot faster.
- **Frame stacking.** The policy gets one instant of encoder ticks; a stack of the last few
  would let it estimate its own velocity rather than inferring it from the goal vector's rate
  of change.
- **Harder odometry.** Widen `ENCODER_NOISE_STD_RANGE`, or add a systematic per-episode wheel
  radius error (real tyres are not all the same diameter), which is the error that actually
  ruins dead reckoning in practice.
- **Curriculum.** Start with zero obstacles and short goals, then scale both up. This task has
  a much sparser success signal than line following, so a from-scratch run may need it.
- **Deployment.** `rover_control/rl_rover.py` already bridges a trained line-follower policy to
  the physical rover. The extra piece here is that the bridge has to run the same
  `integrate_odometry()` against `EncoderPoller.latest()` and feed `lidar_latest()` into the
  observation — the sim was written against those two functions specifically.
