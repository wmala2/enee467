# PPO line-following baseline

This is the next stage after the PID baseline: learn camera-based line following on
2-inch circle, figure-eight, and oval tracks, using the same CAD rover, provisional
camera mount, velocity servos, 10 Hz control rate, and geometric success criteria.
Domain randomization and BAM motor dynamics now have a separate
[integration and evaluation guide](dr-bam-line-follower.md), including parameter ranges
and the physical deployment interface.

The nominal policy now completes **40/40 independent test episodes on each of the three
tracks**, including figure-eight, using the original camera reward.
See [measured results](#measured-nominal-results) for the checkpoint, metrics, and artifacts.

## Run training

From `rover_mujoco`, using the workspace environment:

```sh
# Train on all three closed tracks and log to the authenticated W&B account.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run scripts/train_ppo.py --tracks circle figure8 oval

# Choose a new run directory and an explicit training budget.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run scripts/train_ppo.py --tracks circle figure8 oval --timesteps 150000 --n-envs 6 --output runs/my-ppo-run

# Continue an existing checkpoint with an additional training budget in a separate run.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run scripts/train_ppo.py --tracks circle figure8 oval --resume runs/my-ppo-run/checkpoints/rl_model_49992_steps.zip --timesteps 150000 --output runs/my-ppo-continuation

# Inspect local TensorBoard metrics.
uv run tensorboard --logdir runs/my-ppo-run/tensorboard
```

W&B logging is enabled by default, using project `rover-line-follower` and the active
account's default entity; an authentication failure stops startup.
`--no-wandb` is available for local development checks.
Each run requires a fresh output directory so earlier experiments are preserved.
On resume, the runner applies its configured hyperparameters, including command-line
learning rate and entropy coefficient, to the loaded model and records the checkpoint's
step count as `prior_timesteps`.
Training curves count additional steps from zero for the new run; add `prior_timesteps`
when comparing total training budgets.
The original runner loaded checkpoint hyperparameters after writing the new configuration,
so continuation settings could silently differ from the logged experiment.

## Run evaluation

```sh
# Watch one deterministic traversal per track in the MuJoCo viewer.
uv run scripts/evaluate_ppo.py runs/my-ppo-run/best_model.zip --episodes 1 --viewer

# Save independent test metrics, trajectory plots, and camera videos.
MUJOCO_GL=egl uv run scripts/evaluate_ppo.py runs/my-ppo-run/best_model.zip --episodes 20 --seed 10000 --video

# Export the same scalar curves recorded in TensorBoard and mirrored to W&B.
uv run scripts/plot_ppo_training.py runs/my-ppo-run
```

The registered environment is directly usable from Gymnasium:

```python
import envs
import gymnasium as gym

# Construct the same environment used by the PPO runner.
environment = gym.make("LineFollowerPPO-v0", track="figure8")
observation, info = environment.reset(seed=0)
```

## Small modules with explicit responsibilities

| Module | Responsibility |
|---|---|
| `envs/line_scene.py` | Build the common CAD rover, tape geometry, and provisional camera |
| `envs/camera.py` | Extract near, middle, and far line centroids from RGB pixels |
| `envs/tasks/line_follower_ppo_env.py` | Reset, observe sensors, apply actions, compute reward and outcomes |
| `scripts/train_ppo.py` | Configure SB3, collect parallel rollouts, log, and select checkpoints |
| `scripts/evaluate_ppo.py` | Measure completion separately from reward and record behavior |
| `scripts/plot_ppo_training.py` | Export numerical training curves and a shareable figure |

The previous `LineFollower-v0` and `LineFollowerReal-v0` environments remain separately
registered for their existing models and experiments.

## Observation and action contract

The observation is an 11-element `float32` vector in `[-1, 1]`:

| Indices | Meaning |
|---|---|
| 0–2 | Near-band centroid x, centroid y, and visibility |
| 3–5 | Middle-band centroid x, centroid y, and visibility |
| 6–8 | Far-band centroid x, centroid y, and visibility |
| 9–10 | Left and right wheel-speed estimates, divided by 10 rad/s |

Centroid coordinates are normalized relative to the full 64×64 image.
Bands cover image rows 85–100%, 55–85%, and 25–55%, respectively.
The near band reproduces the PID's extraction exactly; a band needs at least six dark
pixels below RGB threshold 60 to count as visible.
Missing bands contain `[0, 0, 0]`, so a centered visible line remains distinguishable.

Encoder speed is calculated from wheel-angle change over each 0.1-second control
interval, with signs converted so forward rotation is positive on both wheels.
These initial encoders are ideal: quantization, sensor noise, latency, and communication
handling belong to the later hardware-realism stage.
There is no ground-truth position, track identity, yaw, progress, or body velocity in
the policy observation.
All normalization is fixed and documented here; this baseline does not use `VecNormalize`.

The action is `[forward, steering]`, each in `[-1, 1]`:

```text
forward wheel speed = 3 × (forward + 1) rad/s       # 0 to 6 rad/s
steering correction = 4 × steering rad/s           # -4 to +4 rad/s
MuJoCo left command = -forward wheel speed + steering correction
MuJoCo right command = forward wheel speed + steering correction
```

Thus `[0, 0]` reproduces the PID cruise command of approximately 0.103 m/s,
`[-1, 0]` requests a stop, and both wheel commands remain within ±10 rad/s.
The joint-sign mapping is specific to the CAD model; physical deployment must use the
firmware's wheel-velocity convention and units.

## Reward and success

Reward uses ordered ground-truth path progress only as a training signal, never as
policy input:

```text
progress reward = clip(delta arc length / (0.10305 m/s × 0.1 s), -2, 1)
alignment = 1 - abs(near centroid x)
motion reward = progress reward × (0.25 + 0.75 × alignment), or -1 if near line is missing
reward = motion reward - 0.02 × squared action change - 0.01
terminal reward adjustment = +10 for completion, -5 for failure
```

There is no positive alignment reward for simply standing still.
The default `--reward-centering camera` retains the original camera-centering shaping.
The optional `--reward-centering chassis` ablation replaces alignment with
`1 - min(chassis deviation / 0.06 m, 1)`, using the success criterion's ordered path
deviation only for reward.
The evaluator reads this setting from the model's run configuration, with an explicit
command-line override available for models stored separately from their configuration.
Positive progress reward saturates around the PID cruise speed; this stage prioritizes
reliable tracking before increasing speed.
The small action-change penalty encourages smooth commands without penalizing every
necessary turn.

Success requires a complete ordered lap within 3 cm of the starting arc-length position,
no base deviation above 6 cm, no consecutive near-band line loss lasting 0.5 seconds,
and completion within 120 seconds.
Tipping, excessive deviation, and sustained line loss are **terminations**;
the time limit is a **truncation** and is scored as unsuccessful.
The figure-eight evaluator searches only near its previous segment to avoid jumping
between branches at the crossing.
The open S-curve is excluded from this first PPO stage because its endpoint behavior
remains unresolved in the PID baseline.

## Training and checkpoint selection

Defaults use SB3 `MlpPolicy`, two 64-unit hidden layers, and six independent MuJoCo
workers, with two workers assigned to each track.
Training starts at randomly selected waypoints, adding ±1 cm lateral and ±5° heading
perturbations; a one-second settling period precedes every episode.
Physical parameters and camera appearance remain fixed.

| Hyperparameter | Default |
|---|---:|
| Learning rate | 0.0003 |
| Rollout steps per worker | 512 |
| Minibatch size | 256 |
| PPO epochs per rollout | 10 |
| Discount factor | 0.995 |
| GAE lambda | 0.95 |
| PPO clip range | 0.2 |
| Entropy coefficient | 0.005 |
| Value-loss coefficient | 0.5 |
| Maximum gradient norm | 0.5 |
| Initial policy log standard deviation | -1.0 |
| Requested environment steps | 250,000 |

SB3 finishes whole rollouts, so the actual step count may exceed the requested budget.
Every 25,000 environment steps, deterministic evaluation uses five fixed-start episodes
per track with seeds 1000–1004; checkpoints are ranked first by their **worst track's
completion rate**, with mean return breaking ties.
Final testing uses 40 new seeds, 10000–10039, per track.
These tests assess fresh starting perturbations on known track shapes, not generalization
to unseen geometries or physical hardware, and one training seed does not establish
training reliability across seeds.

The run saves the best and final policies, periodic checkpoints, the full configuration,
validation history, TensorBoard events, per-episode test metrics, trajectories, and camera
videos; W&B receives training scalars, test summary metrics, a trajectory image, and the
selected model artifact.

## Checks and sources

```sh
# Check sensor behavior, failure semantics, seed reproducibility, PID parity, and SB3 save/load.
MUJOCO_GL=egl uv run python -m pytest tests/test_pid_baseline.py tests/test_ppo_line_follower.py tests/test_ppo_training.py -q
```

The environment passed SB3's environment checker and the PID completed each closed
track through the new observation/action interface before PPO training began.
The implementation follows the [SB3 PPO API](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
and [Gymnasium environment contract](https://gymnasium.farama.org/introduction/create_custom_env/).

## Figure-eight diagnosis

The interrupted `ppo-baseline-seed0` run contains validation at 25,002 and 50,004 steps,
with 5/5 completions on both circle and oval and 0/5 on figure-eight at each checkpoint.
Its best saved model is from 25,002 steps, and it has no final test report.
That evidence does not establish failure after the configured 150,000-step budget.

For validation seed 1000, the selected model leaves the ordered figure-eight path after
4.1 simulated seconds and 8.5% progress, before reaching the crossing.
Holding PPO's forward action at PID cruise speed still fails after 3.9 seconds.
PID using exactly the PPO environment's observation and action interface completes the
lap in 46.2 seconds with a maximum base deviation of 4.97 cm.
A proportional-only controller also completes seeds 1000, 1001, and 1002, showing that
derivative history is not necessary to solve these nominal episodes.
The regression suite now checks PID completion through this interface.

The comparison traces and plot are stored locally under `artifacts/ppo-diagnosis`.
PPO's near-camera centroid is better centered early in the bend while its chassis path
eventually exceeds the 6 cm deviation limit; camera centering and chassis tracking must
therefore be examined separately.
The controlled continuation retains the reward and environment to test training budget
before changing task design, consistent with the
[SB3 guidance on training budgets and separate evaluation](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

The paired chassis-reward experiment resumes the same 49,992-step checkpoint with the
same seed, hyperparameters, tracks, and 150,000 additional steps.
It replaces only the centering reward with normalized chassis deviation; the tests verify
that identical action sequences give identical observations, physics, and task outcomes
under both reward modes.
New runs also log training completion, deviation, and progress separately for each track.
Completion rates are comparable across reward modes, but shaped episode returns are not.

## Measured nominal results

On 2026-09-09, `ppo-continued-seed0` completed 150,528 additional environment steps after
resuming the original 49,992-step checkpoint, for 200,520 cumulative interactions.
The best policy was selected at 150,000 additional steps, or 199,992 cumulatively.
All five validation episodes per track passed at that checkpoint; evaluation then used
40 reserved seeds per track, 10000–10039, with deterministic policy actions.

| Track | Completed | Success rate | Mean base deviation (cm) | Maximum base deviation (cm) | Mean lap time (s) |
|---|---:|---:|---:|---:|---:|
| Circle | 40/40 | 100% | 2.69 | 2.88 | 21.09 |
| Figure-eight | 40/40 | 100% | 2.08 | 5.68 | 42.29 |
| Oval | 40/40 | 100% | 2.39 | 4.30 | 24.34 |

The result meets the requested **observed success rate of at least 90% on every track**.

The seed count is what lets that statement carry per track. A perfect 20/20 only bounds the
true rate at 86.1% with 95% confidence, which sits under the 90% requirement; 40/40 raises
that bound to 92.8%, and the pooled 120/120 across all three tracks reaches 97.5%. The added
seeds behaved like the original ones: every mean deviation and lap time above differs from the
20-seed measurement by at most 0.01 cm and 0.01 s, so doubling the sample tightened the bound
without moving the estimate.
These are new start-perturbation seeds on known geometries with nominal simulation
physics, from one training seed; DR/BAM and real-world qualification remain separate.
This policy was trained with PPO alone, without PID demonstrations or a PID fallback.
The effective correction was completing a sufficient training budget, not changing the
sensor contract or relaxing the success criteria.
The default budget is now 250,000 steps to allow more training beyond the earlier cutoff,
but each resulting model still requires independent evaluation.

```sh
# View the selected nominal policy using its saved reward configuration.
uv run scripts/evaluate_ppo.py runs/ppo-continued-seed0/best_model.zip --episodes 1 --viewer
```

The local run directory contains `best_model.zip`, `final_model.zip`, `config.json`,
`validation.json`, `training_scalars.csv`, and `training_curves.png`.
Its `evaluation` directory holds every episode's CSV, the full `evaluation.json`, camera
MP4s, a trajectory plot, and `qualification.json` with the checkpoint SHA-256 and observed
success-rate decision. `evaluation-40` holds the 40-seed rerun of the same checkpoint, whose
`qualification.json` records the same SHA-256, so both records refer to one identical model.
The [W&B run](https://wandb.ai/cursedrock17-university-of-maryland/rover-line-follower/runs/hzl1xy2f)
contains the hyperparameters, training and validation curves, test summary, trajectory
image, and selected model artifact.

![Nominal PPO training curves, measured in additional environment steps](images/Resources/ppo_nominal_training.png)

![All 60 nominal PPO evaluation trajectories](images/Resources/ppo_nominal_trajectories.png)

All 15 simulation and training tests pass, including PID feasibility, reward isolation,
and an actual resumed-training CLI run.
Ruff formatting, Ruff lint, and ty pass on the changed Python files.
The repository-wide check still reports the pre-existing guide-formatting issue and
171 type diagnostics outside these changes.

## Downloading the saved policy

See [Download and run saved policies](policy-downloads.md) for the Hugging Face
checkpoint and configuration downloads, a 3D viewer command, and optional Python
loading from the local HF cache.
