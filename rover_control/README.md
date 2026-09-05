# Rover Control
This folder contains the higher-level control code for driving the STEM rovers. It builds on `ArUco_detector`, `YOLO_agent`, and `depth_anything_server` to provide a simple high-level API plus a set of runnable example scripts.

## Setup
If you have already run `pip install -e .` from the repository root, all dependencies (including `pynput` for keyboard control) are already installed. Simply activate the conda environment:

```bash
conda activate rover_high_level
```

## Simple API
The easiest way to drive the rover: create one `Rover` and call high-level methods. No control loops, no PID, no estimator wiring - all of that lives inside these methods. See [examples/simple_api_demo.py](examples/simple_api_demo.py).

```python
from rover_control.rover import Rover

rover = Rover(show_camera=True)        # show_camera=True pops up what the camera sees
try:
    rover.run_maze([0, 1])             # drive to ArUco tag 0, then tag 1
    rover.follow_object("bottle")      # find a COCO object and drive up to it
finally:
    rover.stop()
    rover.close()                      # shuts down the background camera reader
```

| Method | What it does |
|--------|--------------|
| `run_maze([0, 1, 2])` | Drive to a list of ArUco tags in order. |
| `drive_to_tag(0)` | Search for one ArUco tag, center on it, drive up to ~0.25 m away. |
| `follow_object("bottle")` | Find a COCO object (YOLO), center on it, drive up to it. |
| `run_object_maze(["bottle", "cup"])` | Drive to a list of COCO objects in order. |
| `forward(0.5)` | Drive straight 0.5 m on a trapezoidal speed curve (negative = backward). |
| `turn(-90)` | Spin in place; **positive = right**, negative = left (degrees). |
| `see_tag(0)` | One stop-and-stare reading: `{"position": [x, y, z], "bearing": deg, "distance": m}` or `None`. |
| `stop()` / `close()` | Halt the rover / shut down the camera reader. |

**Coordinate frame:** `position` is `[X, Y, Z]` in meters, as seen from the rover - **+X right, +Y up, +Z forward**.

**Good to know:**
- `forward()` and `turn()` are **open-loop** (timed), so distance/angle drift a little; `drive_to_tag`/`follow_object` re-measure visually, so they self-correct.
- Tag/object readings are taken **while stopped** (the rover turns, halts, takes several agreeing frames, then decides) to beat the camera's 1-2 frame lag - reliable but not fast. `follow_object` is slowest (~1 s/frame for the depth step).
- Every motion method **stops the rover on Ctrl-C** or any error, so it can't run away.
- The PID-based and per-frame examples below are the "under-the-hood" versions for students who want to see or extend the internals.

## Example Executables

| Executable | Purpose |
|------------|---------|
| [Simple API Demo](examples/simple_api_demo.py) | The whole student program in a few lines: create a `Rover` and call `run_maze` / `drive_to_tag` / `follow_object`. |
| [Manual Control](examples/manual_control.py) | Drive the physical rover with your keyboard, with adjustable speed scaling. |
| [ArUco Pose Movement](examples/aruco_pose_movement.py) | Autonomously navigate to an ArUco tag and stop a set distance away. |
| [ArUco Tag Tracker](examples/aruco_tag_tracker.py) | Keep tracking a moving ArUco tag at a fixed standoff distance, spinning to find it if lost. |
| [YOLO Object Movement](examples/yolo_object_movement.py) | Navigate the rover toward a named COCO object using YOLO 3D pose, stopping a set distance away. |
| [ArUco Maze Runner](examples/aruco_maze_runner.py) | Drive through a list of ArUco tags in order using a continuous PID approach. |
| [ArUco Maze Runner (Trapezoid)](examples/aruco_maze_runner_trapezoid.py) | Same maze, but measures the tag once then drives a pre-planned trapezoidal speed curve to it. |
| [Encoder Readout](examples/encoder_readout.py) | Poll the rover for its wheel encoder counts over UDP and print the replies. |
| [RL Line Follower](examples/rl_line_follower.py) | Drive using a PPO policy trained in [rover_mujoco](https://huggingface.co/CursedRock17/rover-line-follower-ppo) instead of hand-written control logic — see `rover_control/rl_rover.py` for the sim<->real translation (sign convention, units, real velocity clamping). Never validated on real hardware before this; watch closely on first run. |
| [DA3 Object Follower](examples/da3_object_movement.py) | Follows a named COCO object (default: person) in real time using GPU depth + PID control. |
| [Rover Mission Dashboard](examples/rover_yolo_estimator.py) | Side-by-side live camera + depth view with YOLO overlays; press S to save a depth histogram. |

---

## Design Decisions

### PID approach vs. trapezoidal drive
The **PID maze runner** (`aruco_maze_runner.py`) closes the loop on the tag continuously during the approach — every tick it re-measures the tag's position and adjusts wheel speeds. This works well at low speeds but degrades at high speeds: motion blur and vibration cause the detector to lose the tag mid-approach, triggering repeated recenter-and-retry cycles.

The **trapezoidal maze runner** (`aruco_maze_runner_trapezoid.py`) measures the tag once while stopped, computes the full trajectory upfront, then drives open-loop with a ramp-up / cruise / ramp-down speed profile. Because it does not need the camera during the drive, it is immune to motion blur. The trade-off is that it cannot correct for drift mid-course. In practice the trapezoidal variant is more reliable for classroom use on flat surfaces.

### Stop-and-stare vs. continuous streaming
All pose measurements are taken **while the rover is stopped**. After stopping, the code waits a short settling time and then collects multiple frames, accepting only those captured after the settle window. This discards frames that were in the camera's pipeline during motion. The cost is speed — each measurement takes ~0.2–0.5 s — but the benefit is that every pose the control loop acts on is from a stationary, sharp image. Continuous streaming was evaluated and rejected because the ESP32's 1–2 frame pipeline lag meant the control loop was always acting on slightly stale data.
