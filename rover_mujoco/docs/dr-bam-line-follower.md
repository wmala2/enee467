# DR and BAM line follower

The training and evaluation scripts support three dynamics modes:

- `--dynamics nominal`: the qualified velocity-servo baseline.
- `--dynamics bam`: fixed BAM/DC motor parameters and the firmware speed envelope.
- `--dynamics dr`: BAM with motor, contact, camera, noise, and command-delay
  parameters sampled at every reset.

All three use the CAD rover, 64×64 camera, eleven sensor inputs, normalized
forward/steering actions, and 10 Hz policy rate described in
[the nominal bringup](ppo-line-follower.md).
Completion still requires a full lap within the existing 3 cm finish tolerance,
at most 6 cm chassis deviation, less than 0.5 seconds of continuous line loss,
and at most 120 seconds of simulated time.

## Motor integration and parameter ranges

`envs/line_dynamics.py` defines the resolved parameter ranges stored in every DR
run's `config.json` and W&B configuration.
Continuous parameters are sampled uniformly once per episode; command delay is
sampled uniformly over the integer endpoints, and motor scale factors are drawn
independently for the left and right wheels.
Pixel and encoder noise are zero-mean Gaussian draws at each observation using
the episode's sampled standard deviation.
All draws use the Gymnasium episode seed, and each saved evaluation episode
includes its complete `domain_parameters` dictionary.

These ranges are provisional engineering assumptions; they have not been
established through motor identification or measured confidence intervals.

| Parameter | Minimum | Maximum | Description |
| --- | ---: | ---: | --- |
| `mass_scale` | 0.9 | 1.1 | Multiply each moving body's CAD mass and inertia by this common dimensionless factor. |
| `contact_friction` | 0.6 | 1.2 | Set the dimensionless sliding friction coefficient on both sides of all contacts. |
| `voltage` | 10.8 V | 12.6 V | Limit the voltage available to both wheel motors for the episode. |
| `kt_scale` | 0.9 | 1.1 | Scale each wheel's output-shaft torque and back-EMF constant independently. |
| `resistance_scale` | 0.9 | 1.1 | Scale each wheel's winding resistance independently. |
| `kp_scale` | 0.8 | 1.2 | Scale each wheel's proportional speed-to-voltage gain independently. |
| `friction_scale` | 0.5 | 1.5 | Scale each wheel's BAM Coulomb, Stribeck, and viscous friction terms together. |
| `camera_x_m` | −0.003 m | 0.003 m | Offset the camera laterally from its nominal mount position. |
| `camera_y_m` | −0.003 m | 0.003 m | Offset the camera longitudinally from its nominal mount position. |
| `camera_z_m` | −0.002 m | 0.002 m | Offset the camera vertically from its nominal mount position. |
| `camera_tilt_deg` | 42° | 48° | Vary camera tilt around the nominal 45° mount angle. |
| `camera_fovy_deg` | 57° | 63° | Vary the camera's vertical field of view around 60°. |
| `brightness` | 0.85 | 1.15 | Multiply rendered RGB intensities before extracting the shared camera features. |
| `pixel_noise_std` | 0 | 2 intensity levels | Set the Gaussian pixel-noise standard deviation before clipping to 8-bit RGB. |
| `encoder_noise_std` | 0 rad/s | 0.1 rad/s | Set the Gaussian noise standard deviation added to each wheel-speed observation. |
| `command_delay_steps` | 0 | 1 | Delay wheel targets by zero or one 100 ms policy step using a reset-cleared queue. |

Fixed BAM uses unit scales, contact friction 1, 12 V, the nominal camera mount,
and zero noise or command delay.
Its base electrical constants remain the existing `envs/motor.py` values:
`kt = 0.10787315 Nm/A`, `R = 12 ohm`, and velocity gain
`6 V/(rad/s)`.
Coulomb and Stribeck friction are each 10% of the stated stall torque;
viscous friction is `0.0053936575 Nm/(rad/s)`.
These values are assumptions derived from the datasheet, and the existing
stall-derived electrical model predicts a no-load speed about 2.3 times the
datasheet's stated 463 RPM.
Randomization around those assumptions does not resolve that inconsistency.

The scene is compiled with torque actuators only for BAM modes.
At every 2 ms physics substep, the speed controller computes voltage-limited
drive torque, moves the speed-dependent electrical torque into implicit joint
damping, and applies BAM's Stribeck friction alongside it.
This reuses the motor integration tested against BAM's equation and a free-wheel
steady-state solution.
[BAM provides friction models and identification from recorded trajectories](https://github.com/Rhoban/bam),
so actual response logs are the next input needed to fit this rover's model.

Contact friction is applied to the floor and wheel collision geoms together
because [MuJoCo uses the maximum coefficient for equal-priority contact geoms](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).
Mass and inertia are restored from the compiled baseline before every sample,
then `mj_setConst` refreshes derived model constants.

Both BAM modes clip wheel targets at `0.25 / 0.03435 = 7.278 rad/s` and zero
targets below `0.075 / 0.03435 = 2.183 rad/s` in magnitude.
The linear limits come from `Rover` and use the CAD collision radius.
This path does not use the old motor helper's 33.5 mm radius or `Rover`'s 35 mm
default.
The envelope remains fixed pending measurements of the motor dead zone and
wheel radius.

## Training and independent evaluation

Commands below run from `rover_mujoco` and use EGL for headless rendering.
The original three tracks are explicit so adding another registered track does
not silently change the experiment.

```bash
# Fine-tune a qualified nominal checkpoint under randomized BAM dynamics.
MUJOCO_GL=egl uv run scripts/train_ppo.py \
  --resume runs/ppo-continued-seed0/best_model.zip \
  --dynamics dr --tracks circle figure8 oval \
  --timesteps 100000 --eval-episodes 5 --test-episodes 20 \
  --wandb-group dr-bam-bringup --output runs/ppo-dr-new

# Evaluate fresh randomized episodes with the saved training distribution.
MUJOCO_GL=egl uv run scripts/evaluate_ppo.py runs/ppo-dr-new/best_model.zip \
  --dynamics dr --tracks circle figure8 oval --episodes 50 --seed 40000 \
  --output runs/ppo-dr-new/evaluation-dr

# Evaluate the same weights under fixed BAM parameters.
MUJOCO_GL=egl uv run scripts/evaluate_ppo.py runs/ppo-dr-new/best_model.zip \
  --dynamics bam --tracks circle figure8 oval --episodes 20 --seed 40000 \
  --output runs/ppo-dr-new/evaluation-bam
```

Evaluation automatically recovers dynamics, reward mode, selected tracks, and
resolved DR bounds from the checkpoint's run configuration unless overridden.
It writes per-episode CSV trajectories, `evaluation.json`, a trajectory plot,
and `evaluation_metadata.json` containing the checkpoint SHA-256 and test settings;
`--video` adds camera footage and `--viewer` runs the MuJoCo viewer in real time.
W&B training logs include optimization curves, per-track completion and deviation,
the resolved hyperparameters, and the selected model artifact.

To tune ranges, supply `--dr-ranges ranges.json` with a subset of parameter bounds,
for example `{"command_delay_steps": [0, 0], "kp_scale": [0.9, 1.1]}`.
The run saves the merged ranges, preserving its evaluation distribution if the
input file changes later.
Compare candidates on fixed validation seeds and reserve fresh seeds for final
testing; any test set used to guide training becomes part of selection.

## First DR training result

The experiment is saved under `runs/dr-bam-bringup`, with a frozen source and
asset snapshot and SHA-256 manifest for the training run.
It starts from `ppo-continued-seed0/best_model.zip`, uses six workers, camera
centering reward, learning rate 0.0003, entropy coefficient 0.005, gamma 0.995,
512 steps per rollout per worker, and a `[64, 64]` MLP.
The requested additional budget was 100,000 steps; SB3 collected 101,376 to
finish its last rollout, and the selected checkpoint was evaluated at 100,002.

Before fine-tuning, the original policy scored 5/5 circle, 0/5 figure eight,
and 5/5 oval under both fixed and randomized BAM, using seeds 30000–30004.
This demonstrated a transfer gap despite nominal qualification.

| Additional steps | Circle validation | Figure-eight validation | Oval validation |
| ---: | ---: | ---: | ---: |
| 0 | 5/5 | 0/5 | 5/5 |
| 25,002 | 5/5 | 3/5 | 5/5 |
| 50,004 | 5/5 | 1/5 | 5/5 |
| 75,000 | 5/5 | 2/5 | 5/5 |
| 100,002 | 5/5 | 5/5 | 5/5 |

Selection ranks the weakest track's completion first, then mean validation
return, using seeds 1000–1004 throughout.
The selected policy completed **20/20 on each track** under DR with the independent
seeds 10000–10019; figure-eight mean deviation was 1.88 cm and its maximum was
5.63 cm.
The [W&B training run](https://wandb.ai/cursedrock17-university-of-maryland/rover-line-follower/runs/dc0dldgt)
contains the corresponding hyperparameters, learning curves, test summaries,
and model artifact.

![DR training and validation curves](images/Resources/ppo_dr_training.png)

The final tests below use seeds 40000 onward on the original tracks and
50000–50019 on Goomba, with deterministic policy actions and no further selection.

| Policy and test conditions | Track | Completions | Mean deviation | Maximum deviation | Mean lap time |
| --- | --- | ---: | ---: | ---: | ---: |
| Nominal reference, velocity servos | Circle | 20/20 | 2.71 cm | 2.88 cm | 21.08 s |
| Nominal reference, velocity servos | Figure eight | 20/20 | 2.08 cm | 5.68 cm | 42.29 s |
| Nominal reference, velocity servos | Oval | 20/20 | 2.40 cm | 4.30 cm | 24.33 s |
| DR policy, fixed BAM | Circle | 20/20 | 3.14 cm | 3.34 cm | 23.98 s |
| DR policy, fixed BAM | Figure eight | 20/20 | 1.85 cm | 5.25 cm | 46.54 s |
| DR policy, fixed BAM | Oval | 20/20 | 2.72 cm | 4.45 cm | 27.44 s |
| DR policy, randomized BAM | Circle | 50/50 | 3.16 cm | 4.30 cm | 24.18 s |
| DR policy, randomized BAM | Figure eight | 50/50 | 1.90 cm | 5.80 cm | 46.90 s |
| DR policy, randomized BAM | Oval | 50/50 | 2.74 cm | 5.59 cm | 27.70 s |
| DR policy, randomized BAM, unseen geometry | Goomba | 20/20 | 1.92 cm | 4.25 cm | 32.09 s |

Both policies pass the empirical 90% simulation gate on each original track;
the DR policy also passes the held-out Goomba test.
DR produces longer laps under harder dynamics, with lower mean figure-eight
deviation but no uniform improvement across tracks.
These measurements cover one training seed and a provisional simulation
distribution; the physical demonstration is described below.

![Completion and tracking before and after DR fine-tuning](images/Resources/ppo_dr_comparison.png)

The pre-training figure-eight traces terminate early, so their lower mean error
is not evidence of better tracking over a complete lap.

![All 150 fresh DR trajectories](images/Resources/ppo_dr_trajectories.png)

The selected model is `runs/dr-bam-bringup/ppo-dr-seed0/best_model.zip`, SHA-256
`06f837b4cfafbd756c62ae0f97875e6832431da1616b3c2006a64d5fbb2d01bf`.
`runs/dr-bam-bringup/summary.json` contains the comparison metrics, and
`deployment.json` records the successfully verified simulation gate.
Environment and asset hashes used for final evaluation match the training snapshot.
The [W&B qualification run](https://wandb.ai/cursedrock17-university-of-maryland/rover-line-follower/runs/9mh560zs)
contains the final comparison table, trajectory plots, checkpoint, configuration,
and simulation evidence as the `rover-dr-bam-qualified-seed0` model artifact.

## Preparing physical rollout

`rover_control.rl_rover` takes a local checkpoint and deployment manifest;
it does not download the old image-based policy.
It uses the shared eleven-input observation builder and maps normalized actions
to left/right rad/s with both physical wheel signs positive forward.
Only MuJoCo negates the left wheel to match the CAD axle orientation.
Encoder deltas are converted using measured elapsed time, counts per output-shaft
revolution, and explicitly supplied encoder signs.
Camera frames are resized to 64×64 and converted from BGR to RGB before the same
feature extraction used in simulation.

From the workspace root, the following offline command checks the exact model
hashes and per-episode results before writing a manifest:

```bash
# Require at least 90% completion on every original track for both RL variants.
uv run python -m rover_control.ppo_deployment \
  rover_mujoco/runs/dr-bam-bringup/ppo-dr-seed0/best_model.zip \
  --nominal-model rover_mujoco/runs/ppo-continued-seed0/best_model.zip \
  --nominal-evaluation rover_mujoco/runs/dr-bam-bringup/evaluation-nominal-20 \
  --bam-evaluation rover_mujoco/runs/dr-bam-bringup/evaluation-bam-20 \
  --dr-evaluation rover_mujoco/runs/dr-bam-bringup/evaluation-dr-50 \
  --output rover_mujoco/runs/dr-bam-bringup/deployment.json
```

The gate checks geometric completion over at least 20 distinct seeds per original
track; pooled rewards or reported aggregate rates do not satisfy it.
It checks the nominal reference under nominal physics and the exact deployment
checkpoint in both BAM modes.
Goomba is reported separately from this three-track gate.
The manifest is local evidence, not a cryptographic attestation of hardware safety.

The selected policy subsequently ran successfully on the physical rover, as
reported by the experimenter; the [Zero to Hero tutorial](<Deploying a Mini Claw Rover Policy _ Zero to Hero.md#8-deploy-the-selected-policy-on-the-physical-rover>)
records the hardware stage and its differences from simulation.
The entry point is `uv run python -m rover_control.examples.rl_line_follower MODEL MANIFEST`.
It defaults to the firmware profile's 680 encoder counts per wheel revolution
and signs `(1, 1)`, with `--counts-per-revolution` and `--encoder-signs` overrides
for a different profile or calibration.
Use `--wheel-radius-m`, `--camera-addr`, and `--rover-addr` for the physical setup,
and `--record DIRECTORY` to save the camera overlay and observation CSV.

The current runner polls the camera at 12.5 Hz, sends zero during sensor startup,
and stops on sensor data older than 1.0 s or 15 consecutive missing-line
observations (1.5 s at the 10 Hz policy rate).
It also sends zero on inference errors or keyboard interruption.
These hardware limits replaced the initial 250 ms sensor-age and five-step
line-loss limits after physical bringup; simulation still uses 0.5 s of line loss.
The successful physical demonstration does not establish a hardware lap-success
rate or make the provisional motor parameters a measured fit.

## Verification

At DR bringup, the full simulation suite passed 42 tests plus a targeted
runtime-parity test, covering nominal behavior, seeded DR replay after intervening
resets, contact coefficients, delayed motor commands,
the motor equation and free-wheel integration, encoder units, deployment-gate
rejections, and stop behavior without hardware access.
Run it from `rover_mujoco` with
`MUJOCO_GL=egl uv run python -m pytest tests -q`.
At that stage, repository Ruff formatting and lint passed; the root
`uv run scripts/check.py` reported 124 existing type diagnostics, including three in
`envs/motor.py` BAM setup involving its optional testbench and dynamic attributes.
The new modules, runner, environment, training/evaluation scripts, and tests
passed scoped `ty` checks.
These counts record the bringup checks; rerun the commands for the current checkout.

## Downloading the saved policy

See [Download and run saved policies](policy-downloads.md) for the Hugging Face
checkpoint and configuration downloads, a 3D viewer command, and optional Python
loading from the local HF cache.
