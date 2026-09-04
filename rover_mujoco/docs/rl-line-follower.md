# RL line follower: sim-only, then sim-to-real-constrained

Two environments, registered as `LineFollower-v0` and `LineFollowerReal-v0`. Both train a
policy to follow the black line tracks from `docs/pid-line-follower.md` using PPO instead of
that tutorial's PID/bang-bang controllers. The difference between them is the whole point of
this doc: `LineFollower-v0` is a reasonable first RL environment; `LineFollowerReal-v0` is
what it takes to make that policy's training conditions actually resemble deploying it on the
physical rover.

## What to experiment with (either environment)

This isn't a "run it once and you're done" tutorial — the curriculum goal is comparing your
own choices against the working baseline below:

- **Algorithms.** `stable-baselines3` also ships A2C, and DQN if you discretize the
  continuous left/right wheel velocity action space into a fixed set of choices (DQN needs a
  discrete action space; PPO/A2C don't). Swapping `PPO` for `A2C` in `train.py`/`train_real.py`
  is a one-line change; compare sample efficiency and final reward.
- **Observation space.** Grayscale vs. color, the current `64×64` resolution vs. smaller/larger,
  a stack of the last few frames instead of one. `LineFollowerRealEnv` also adds wheel
  encoders to the mix — try training without them to see how much they actually help.
- **Reward function.** The current reward (`1 - abs(error)` when the line is visible, `-1`
  when it isn't) is one reasonable choice, not the only one. Try penalizing angular velocity
  (smoother driving), rewarding distance traveled along the track, or shaping it differently
  near the edges of the frame.
- **Hyperparameters.** `n_steps`, `batch_size`, learning rate, and the PPO clip range all
  matter more once you're past a smoke test — see `uv run tensorboard --logdir runs/...` for
  training curves to compare runs against.

## LineFollower-v0: sim-only baseline

Already covered by what `envs/tasks/line_follower_env.py` does: camera-only observations, no
privileged position in the reward, and domain randomization over floor/line color, lighting,
and camera mount pose. See the "sim-to-real aware" reasoning in this doc's git history or
`envs/tasks/line_follower_env.py`'s own comments — the short version is that even the "simple"
environment avoids the most common ways a simulated vision policy fails before reaching
hardware. Run it with `uv run python scripts/train.py` / `scripts/evaluate.py`.

## LineFollowerReal-v0: constrained to what the rover can actually do

`envs/tasks/line_follower_real_env.py` subclasses `LineFollowerEnv` and cuts the environment
down to what's real:

- **Observation**: the onboard camera image, refreshed only at **10 Hz** (matching the real
  rover's `\capture` endpoint rate) rather than every physics step, plus noisy left/right wheel
  encoder readings. No ground-truth position, ever — same as the sim-only env, but now the
  camera itself is rate-limited too, which changes what the policy can react to.
- **Action**: left/right wheel velocity, same shape as `LineFollower-v0`. This is an
  **assumption**, not a confirmed fact — it's not yet been reconciled against the actual
  command format in the rover-firmware repo (units, scaling, whether it's velocity or a raw
  PWM duty cycle). Everything else in this environment (motor lag, latency, noise) layers on
  top of whatever that interface turns out to be, so it's an isolated thing to fix later
  rather than something the rest of the DR depends on.

### Motor model: BAM + the JGA25-371 datasheet

The wheel joints in `assets/robots/rover/*_real.xml` take raw torque, not a MuJoCo
`<velocity>` servo — the torque itself comes from `envs/motor.py`, a DC-motor model of the
rover's actual drive motor,
[JGA25-371, 463 RPM @ 12V](https://www.openimpulse.com/blog/products-page/25d-gearmotors/jga25-371-dc-gearmotor-encoder-463-rpm-12-v-2/),
built on [BAM](https://github.com/Rhoban/bam)'s friction-model library and the paper it's
based on,
["Extended Friction Models for the Physics Simulation of Servo Actuators"](https://scholar.google.com/scholar?q=Extended+Friction+Models+for+the+Physics+Simulation+of+Servo+Actuators).

A few things worth knowing before trusting this model, all documented in more depth in
`envs/motor.py`'s own comments:

- **BAM's `Actuator`/`MujocoController` classes aren't used directly.** They're built for
  *position*-controlled servos (Dynamixel/Feetech-style: command a target angle, PID to it).
  Our wheels spin continuously under *velocity* control — there's no meaningful "target
  angle" for a driving wheel. We use BAM's `Model` (the Stribeck friction budget) directly,
  paired with our own velocity-tracking control law and the same
  `tau = kt*V/R - kt^2*omega/R` DC-motor equation BAM's own actuators use internally.
- **`kt`/`R` are derived from two datasheet numbers (stall torque, stall current), not
  bench-measured.** Plugging them back into the free-run-speed equation predicts roughly
  2x the datasheet's stated 463 RPM — a real, visible discrepancy, and expected: cheap
  gearmotor datasheets are commonly rounded rather than lab-measured. This model is
  "roughly the right order of magnitude," not calibrated.
- **BAM's own default friction parameters are wrong for this motor by ~10x — literally
  larger than the motor's entire stall torque, which would make the wheel physically unable
  to move.** They're tuned for the much larger reference actuators BAM ships fitted models
  for. `envs/motor.py` scales them down to a small fraction of the JGA25-371's own stall
  torque instead — still a guess, not a fit, but at least dimensionally sane.
- **The real fix for all three points above is BAM's identification pipeline**: recording
  actual trajectories from the physical motor and fitting a model to them, rather than
  deriving one from a two-line datasheet. That needs hardware we don't have yet.

### Domain randomization

Each `reset()` draws new values for the parameters below, on top of `LineFollower-v0`'s
existing visual randomization (floor/line color, lighting, camera pose) and the motor model
above. These ranges are **starting points**, not measured hardware values — real
characterization (a proper BAM fit, per-surface friction measurements, and the actual
camera's datasheet) should replace them over time.

| Parameter | Min | Max | Why |
|---|---|---|---|
| Action noise (rad/s, Gaussian) | 0.0 | 0.3 | Per-step jitter on the commanded wheel velocity, approximating real actuator command noise. |
| Action latency (control steps) | 0 | 2 | The rover's 10 Hz control loop introduces communication/processing delay between a decision and the wheels actually responding. Drawn once per episode, which is also standing in for per-agent lag (manufacturing variance between individual rovers) rather than modeling that as a separate parameter — reconsider this if the two turn out to need different distributions. |
| Wheel-floor friction coefficient | 0.4 | 1.2 | Stands in for different floor materials (grass, concrete, carpet) — MuJoCo resolves contact friction from the wheel/floor geom pair, so randomizing the wheel side sweeps the effective range. Distinct from the motor's own internal (gearbox/brush) friction above, which comes from the BAM model instead. |
| Encoder noise (rad/s, Gaussian) | 0.0 | 0.1 | Real encoder measurement/quantization noise on the velocity reading. |
| Camera capture rate (Hz) | 10 (fixed) | 10 (fixed) | Not randomized — this is a known real constraint (the `\capture` endpoint), modeled directly rather than swept. |
| Camera FOV (deg) | 50 | 70 | Lens FOV manufacturing tolerance and mounting variance around `rover.xml`'s `top_cam` default. |
| Brightness (pixel-value multiplier) | 0.6 | 1.4 | Exposure/ambient-light variation, applied after rendering rather than by moving light sources — cheaper and gets the same effect. |
| White balance (per-RGB-channel gain) | 0.85 | 1.15 | Color-temperature variation, applied to the rendered RGB frame before it's collapsed to grayscale (after that point there's no separate color channel left for a tint to act on). |
| Sensor noise (Gaussian, 0-255 scale) | 0.0 | 15.0 | Stands in for JPEG-quality/compression artifacts. Real JPEG noise is structured block artifacts, not i.i.d. Gaussian — this is a simplification, not a faithful compression model, but still forces the policy not to trust exact pixel values. |
| Effective resolution (px) | 16 | 64 (full) | The observation's actual array shape has to stay fixed for SB3, so "resolution" is approximated by pixelating a fixed-size render (block-average down, nearest-neighbor back up) rather than truly changing the rendered size. |

Not yet implemented: a true JPEG-compression model (block-DCT-style artifacts) instead of the
Gaussian-noise stand-in above, and battery discharge (noted in the original curriculum spec as
likely irrelevant — the rovers have run 10 continuous hours on their 3S battery).

### Running it

```shell
uv run python scripts/train_real.py      # saves to runs/ppo_line_follower_real/
uv run python scripts/evaluate_real.py   # opens the viewer, runs the trained policy
```

Dict observations (image + encoders) need SB3's `MultiInputPolicy`, not `CnnPolicy` — already
set up in `train_real.py`. Verified end-to-end with a 512-timestep smoke test (train → save →
reload → predict) before any real training run, same as the other training scripts in this repo.

### What this still doesn't get right

Domain randomization narrows the sim-to-real gap; it doesn't close it. None of this has been
validated against the real rover yet — that's the actual next step once hardware is available,
and it's normal for the first attempt to fail in some specific, informative way (a DR parameter
that needed a wider range, an observation the policy leaned on too heavily, timing that's off
in a way the sim didn't capture). Budget for that iteration rather than expecting the first
sim-trained policy to just work.
