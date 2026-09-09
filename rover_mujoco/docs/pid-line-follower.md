# PID line-following bring-up

The baseline runs the CAD rover with MuJoCo velocity servos, camera-only steering,
and a 10 Hz command loop; physics advances at the scene's 2 ms timestep.
`envs/pid.py` contains the controller and centroid extraction, while
`scripts/line_follower.py` handles simulation and evaluation.

## Run

From `rover_mujoco`, with the workspace environment installed:

```sh
# Watch the rover and the exact camera pixels used for steering.
uv run scripts/line_follower.py --track figure8 --camera

# Evaluate fresh initial offsets without opening a viewer.
MUJOCO_GL=egl uv run scripts/line_follower.py --headless --track all --episodes 20 --seed 100
```

`MUJOCO_GL=egl` selects offscreen rendering on supported Linux systems; headless
execution still renders the onboard camera.
Available tracks are `circle`, `figure8`, `oval`, and `s_curve`.
The runner builds continuous black lines from the shared waypoint functions, with
width **0.0508 m (2 inches)**; the figure eight is a generated path with a crossing,
replacing the old mesh-based baseline track selection.
The runner constructs its scene in memory without writing the shared RL asset files.

The default output folder is `artifacts/pid/<timestamp>` and contains the exact
controller configuration, JSON episode outcomes, CSV trajectories, initial camera
overlays, and a trajectory comparison plot.
Use a fresh `--output` directory for each experiment to preserve previous results.

## Shared track assets

`scripts/line_follower.py` builds its scene through `gen_track.py`'s `build_track_xml`
and `envs/tracks.py`'s waypoint functions, both shared with the RL environments.
Three changes to those shared files landed alongside this baseline and affect what the
runner renders.

The waypoint functions now default to 144 segments per shape, up from 48
(`figure8_waypoints` is unchanged at 160).
Tracks are polylines of flat boxes, so the segment count sets how closely they follow
the analytic curve.
At 48 segments the worst heading step between neighboring boxes was 11.2 degrees, which
put the centerline 1.29 mm off the true curve and pushed each box corner 1.47 mm past
its neighbor on the outside of a bend.
At 144 those figures are 3.9 degrees, 0.14 mm, and 0.51 mm: under 2% of the 2-inch tape
width, and finer than the 64×64 camera resolves.
The serration only ever added area on the outside of a curve, so it biased the centroid
outward on exactly the frames where steering matters.

Track segments are emitted in geom group 2 rather than 3.
MuJoCo's default `MjvOption` mask is `[1, 1, 1, 0, 0, 0]`, so group 3 was invisible to
every renderer unless each call site re-enabled it.
Group 2 renders by default, which is why the `--camera` diagnostic view and the viewer
both show the line without any scene-option handling.

Segments also carry names, `line_0`, `line_1`, and so on.
The RL environments' appearance randomization selects them by name instead of by geom
group, which is what allowed the group number to change at all.

The line width differs by consumer, deliberately.
This runner passes 0.0508 m (2 inches) to match physical tape, while `gen_track.py`'s
default, used for the RL scene assets in `assets/objects/tracks/`, stays at 0.03 m.
A policy trained against the RL assets sees a line about 40% narrower than this baseline
does, which scales the centroid error the two consume.
Aligning them is outstanding work before any sim-to-real comparison.

## Controller and measurements

The controller samples the bottom 15% of a 64×64 RGB image, thresholds dark pixels,
and normalizes the horizontal centroid around the true pixel center.
A missing line is represented by `None`, distinct from a centered line with zero error.
Detections require at least six dark pixels, rejecting an isolated caster pixel that
otherwise made a blank floor appear to contain a centered line.
Red pixels in the diagnostic view are the detections; green marks the centroid.

Defaults are `--kp 2 --ki 0 --kd 0.1 --speed 3`, where speed is the forward
wheel-speed command in rad/s (about 0.10 m/s from the wheel radius).
The PID uses elapsed seconds for both integral and derivative, suppresses derivative
kick on initialization/reacquisition, and freezes integration when error drives further
into saturation.
With zero integral gain the default is a **PD setting of the PID controller**;
a nonzero integral term is available for investigating measured persistent bias.
Steering is limited so both wheel commands remain within ±10 rad/s.
For the CAD joint signs, forward motion is `[-speed, +speed]` and the same steering
correction is added to both commands.

The last steering command is held through brief camera dropouts; 0.5 seconds of
consecutive missing detections stops the command and fails the episode.
`--controller bang_bang` retains a simple comparison controller.

## Camera assumptions

The existing XML camera settles approximately 15.4 cm above the ground.
For this PID experiment, the runner sets a **45-degree tilt from straight down**
and places the lens **5 cm above the CAD baseplate top**, at local
`[0, -0.108, 0.0256413]` m; it settles approximately 11.0 cm above the ground.
The extra 1 cm forward shift from the old mount is provisional because the physical
forward displacement was described qualitatively rather than measured.
The model's travel direction is local **−Y**, so moving the camera forward decreases Y.

The existing 60-degree vertical field of view and square image are retained as
assumptions; the reported 2.75-inch viewing distance has not yet been matched to a
particular image row or image edge.
The CAD plate settles near 6 cm above ground, slightly different from the height
inferred from the approximate physical measurements.
`--camera-tilt` changes orientation for diagnostics while holding lens position fixed;
it does not model the physical mount's movement when tilted.
These runs establish a simulation baseline, not a calibrated hardware camera model.

## Evaluation contract

Each seed adds an initial lateral offset uniformly within ±1 cm and heading offset
within ±5 degrees, then lets the chassis settle for one simulated second.
All episodes begin at the same end/start of each track and traverse the same direction;
these are small spawn perturbations, not domain randomization or arbitrary recovery tests.

Success requires all of the following:

- Advance to within 3 cm of a full ordered lap or open-path traversal.
- Keep the base origin within **6 cm** of the corresponding track segment throughout.
- Avoid losing the line for 0.5 seconds consecutively.
- Finish within 120 simulated seconds.

The 6 cm criterion is a bring-up tolerance, not a claim that the base origin always
stays inside the tape's 2.54 cm half-width.
The evaluator projects onto a small neighborhood of the previous segment and unwraps
closed-loop arc length, preventing the figure-eight crossing from being mistaken for
a shortcut to the other branch.
Ground-truth position is used only for scoring, never as controller input.
Failures and their deviations are retained in the results, and timeouts are failures.

A zero-steering negative control (`--track figure8 --kp 0 --ki 0 --kd 0`)
failed from line loss at 5.6% progress, showing that completion requires
steering rather than a permissive progress score.

The open S-curve currently stops from line loss near its endpoint: the forward camera
runs out of tape before the base reaches the evaluator's 3 cm finish tolerance.
This is recorded as a failure; reaching most of the path does not count as completion.
The circle, figure eight, and oval provide the three closed tracks for the initial
line-following baseline; endpoint recognition is a separate behavior to resolve before
using open paths as a success benchmark.

## Measured baseline (2026-09-09)

Validation used seeds 100–119, the defaults above, and 6 cm maximum base deviation.
Results and raw trajectories are in `artifacts/pid/validation`; the no-steering control
is in `artifacts/pid/no-steering`.

| Track | Completed | Mean base error | Worst base error |
|---|---:|---:|---:|
| circle | 20/20 | 2.23 cm | 2.69 cm |
| figure8 | 20/20 | 1.84 cm | 5.01 cm |
| oval | 20/20 | 1.97 cm | 4.07 cm |
| s_curve | 0/20 | 1.88 cm | 4.17 cm |

Re-running the same seeds after the track resolution went to 144 segments reproduced
every figure in this table to two decimal places, so the baseline covers the current
geometry; that run is in `artifacts/pid/revalidation-n144`.

Errors include all episodes, including incomplete S-curve trials.
The three closed tracks had zero line-loss frames across all 60 trials.
These empirical results cover small initial perturbations under fixed simulation dynamics.
Seven regression tests, Ruff lint/format, and ty passed; the viewer and camera windows
also passed a short smoke run under Xvfb.

## Verification

```sh
# Check control timing math, saturation, missing detections, crossing projection, and scene geometry.
MUJOCO_GL=egl uv run python -m pytest tests/test_pid_baseline.py -q

# Check only the files changed for this baseline.
uv run ruff check envs/pid.py envs/tracks.py scripts/gen_track.py scripts/line_follower.py tests/test_pid_baseline.py
uv run ruff format --check envs/pid.py envs/tracks.py scripts/gen_track.py scripts/line_follower.py tests/test_pid_baseline.py
uv run ty check --python ../.venv/bin/python envs/pid.py envs/tracks.py scripts/gen_track.py scripts/line_follower.py tests/test_pid_baseline.py
```

MuJoCo's compiled exports lack type stubs in this installation, so the runner uses
narrow `ty` suppressions on those binding calls, which are exercised by scene compilation
and the headless runs.
The rendering and stepping API follows the [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html).

Before RL integration, compare simulated and physical camera frames, measure the
mount's forward offset and actual camera field of view, and decide whether to tighten
centering tolerance or require multiple consecutive laps.
