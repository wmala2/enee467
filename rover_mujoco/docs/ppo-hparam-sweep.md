# Nominal PPO hyperparameter sweep

**Completed:** four candidates, with the selected reference continuation passing all
150 fresh test episodes across circle, figure-eight, and oval.

This sweep fine-tunes the qualified nominal camera-reward policy, which completed 20/20
reserved episodes on circle, figure-eight, and oval.
It measures whether further PPO updates improve that policy while preserving completion.
It is a continuation experiment from one training seed, not a comparison of training from
scratch or a DR/BAM experiment.

The preceding reward ablation used the same starting checkpoint and 150,000-step request
for each run, with the following reserved-seed results:

| Reward | Circle | Figure-eight | Oval | Figure-eight mean deviation (cm) |
|---|---:|---:|---:|---:|
| Camera | 20/20 | 20/20 | 20/20 | 2.08 |
| Chassis | 20/20 | 4/20 | 20/20 | 1.74 |

Chassis centering remains an opt-in ablation; its lower mean deviation includes failed
episodes and does not establish reliable track completion.
The [camera run](https://wandb.ai/cursedrock17-university-of-maryland/rover-line-follower/runs/hzl1xy2f)
is the qualified starting point for this sweep, and the
[chassis run](https://wandb.ai/cursedrock17-university-of-maryland/rover-line-follower/runs/ixa9l8vr)
retains its separate artifacts.

![Figure-eight validation under the two centering rewards](images/Resources/ppo_reward_comparison.png)

## Fixed protocol

Every candidate starts from `runs/ppo-continued-seed0/best_model.zip`, representing
199,992 cumulative environment interactions, and requests 50,000 additional steps with
six workers and training seed 0.
SB3 completes whole rollouts, making the actual additional budget 52,224 steps.
The camera reward, three-track mixture, action/observation contract, and all other PPO
hyperparameters remain fixed.
The sweep snapshots its code, assets, checkpoint, and checkpoint lineage before starting;
later workspace edits cannot change one candidate's environment relative to another's.

| Candidate | Changed parameter | Reference value | Candidate value | Description |
|---|---|---:|---:|---|
| `reference` | None | — | — | Continue with the qualified run's original hyperparameters as a matched-budget control. |
| `learning_rate` | Learning rate | 0.0003 | 0.001 | Test larger optimizer updates while keeping the PPO clipping rule unchanged. |
| `entropy` | Entropy coefficient | 0.005 | 0.0 | Remove the explicit entropy bonus while retaining stochastic rollout actions. |
| `discount` | Discount factor | 0.995 | 0.98 | Give distant rewards less weight in return and advantage estimation. |

Each candidate validates the incoming checkpoint at step zero and again every 25,000
additional steps, using five episodes per track with seeds 1000–1004.
The best checkpoint ranks first by its worst track's completion rate, then by average
validation return; the incoming checkpoint remains eligible if fine-tuning degrades it.
The same validation rule selects the sweep winner.

Candidate evaluation on seeds 10000–10019 is retained as a common benchmark and is not
used for selection.
Only after selection, the winner receives 50 fresh episodes per track on seeds
20000–20049, with deterministic actions and the same failure criteria.
An observed success rate of at least 90% on **every** track is required for that final
test to pass.
The newer Goomba track is excluded from this controlled three-track comparison.

## Commands and artifacts

Run from `rover_mujoco`:

```sh
# Run the four matched continuations, log them to W&B, and independently test the winner.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run scripts/sweep_ppo.py runs/ppo-continued-seed0/best_model.zip --output runs/ppo-hparam-sweep-seed0
```

The sweep writes a root configuration and `source_manifest.json` before training.
Each candidate has its own output directory, W&B training run, log, full configuration,
validation history, best/final policies, checkpoints, numerical training scalars,
training figure, evaluation CSVs, trajectories, and camera videos.
`summary.json` records completed candidates, and `winner.json` records the selected
checkpoint's SHA-256, selection rule, fresh test rates, and pass/fail result.
`final_evaluation` holds the winner's independent test artifacts.
Training and final evaluation runs share the W&B group `ppo-hparam-sweep-seed0`.

The training CLI now accepts `--gamma` and `--tracks` for reproducible comparisons.
The evaluator defaults to the saved configuration's tracks, so adding a registered
track does not silently change an older checkpoint's evaluation protocol.
Explicit `--tracks` can be used to evaluate additional geometries separately.
`prior_timesteps` includes earlier continuations' cumulative interactions, while
`checkpoint_timesteps` records SB3's local counter from the loaded checkpoint.

These parameters follow the [SB3 PPO API](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html),
with separate validation and final testing following the
[SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

## Candidate results

All selected candidate checkpoints completed 20/20 common benchmark episodes on each
track; selection used the separate validation data, with the results below.
The selected-step column counts interactions added after the qualified starting policy.

| Candidate | Selected additional steps | Mean validation return | Figure-eight mean deviation (cm) | Figure-eight mean lap time (s) |
|---|---:|---:|---:|---:|
| Original qualified policy | 0 | 241.05 | 2.08 | 42.28 |
| Reference continuation | 25,002 | **264.68** | 2.29 | 43.32 |
| Learning rate 0.001 | 50,004 | 263.65 | 2.28 | 42.70 |
| Entropy coefficient 0 | 0 | 241.05 | 2.08 | 42.28 |
| Discount factor 0.98 | 25,002 | 257.85 | 2.32 | **39.32** |

The reference continuation won by the declared validation score while retaining the
original learning rate, entropy coefficient, and discount factor.
It regressed to 0/5 figure-eight validation completions at the final training checkpoint,
making preservation of the earlier selected model essential.
The entropy candidate also regressed and retained its incoming policy; its benchmark
repeats the original policy's same 20 cases and is not evidence of a learned improvement.

The original qualified policy had the lowest mean figure-eight chassis deviation among
these distinct policies.
The shorter discount horizon completed figure-eight laps about 7.0% faster than that
original policy, with about 11.6% greater mean deviation.
Thus the validation-return winner, the most accurate tracker, and the fastest policy
are different choices under this reward and evaluation protocol.
The original remains a useful accuracy control for the next DR/BAM stage, and the
shorter-horizon candidate provides a measured speed tradeoff.

![Figure-eight validation and selected checkpoints for all four candidates](images/Resources/ppo_sweep_validation.png)

![Common-seed figure-eight tracking and lap-time distributions](images/Resources/ppo_sweep_tracking.png)

The numerical comparison is saved as `candidate_comparison.csv` in the sweep directory.

## Winner's fresh evaluation

The selected reference continuation passed all 50 episodes per track on seeds
20000–20049, which were excluded from candidate selection.
Its checkpoint represents 224,994 cumulative environment interactions.

| Track | Completed | Success rate | Mean base deviation (cm) | Maximum base deviation (cm) | Mean lap time (s) |
|---|---:|---:|---:|---:|---:|
| Circle | 50/50 | 100% | 3.50 | 3.74 | 21.27 |
| Figure-eight | 50/50 | 100% | 2.29 | 5.71 | 43.31 |
| Oval | 50/50 | 100% | 3.06 | 5.19 | 24.73 |

The fresh result passes the requested observed 90% success threshold on every benchmark
track, under nominal simulation conditions.
The [final W&B evaluation](https://wandb.ai/cursedrock17-university-of-maryland/rover-line-follower/runs/g09tliwh)
contains these metrics, trajectories, and the selected-model artifact; the
[reference training run](https://wandb.ai/cursedrock17-university-of-maryland/rover-line-follower/runs/rcq1b6au)
contains the hyperparameters and training curves.

```sh
# View the sweep's selected checkpoint on its saved three-track protocol.
uv run scripts/evaluate_ppo.py runs/ppo-hparam-sweep-seed0/reference/best_model.zip --episodes 1 --viewer
```

The selected model SHA-256 is
`648c6764241380cc2464e71b42585a0a7ac940cc9808deb28c1c55d85a006163`.
The final audit verified every snapshot-file hash, matched all four saved models to
their recorded selected steps, and checked all fresh episode seeds and success counts.
The 30-test suite passes from `rover_mujoco`; changed Python files pass Ruff and ty,
and repository-wide formatting and lint pass.
The full repository type check still reports 124 diagnostics outside these changes.

## Downloading the saved policy

See [Download and run saved policies](policy-downloads.md) for the Hugging Face
checkpoint and configuration downloads, a 3D viewer command, and optional Python
loading from the local HF cache.
