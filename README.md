# Matrix Lab Rover — High Level
This repository contains the higher-level AI agent code for the STEM Rovers: object detection, ArUco pose estimation, LLM integration, and the control loops that tie them together into autonomous behaviours.
The intent is to elevate the basic differential drive rover platform we're currently control with closed loop techniques, with modern neural network, higher level algorithms.
This project is a joint effort between [UMD's ECE department](https://ece.umd.edu/) and [MATRIX Lab](https://matrix.umd.edu/)

## Installation

### Prerequisites
1. Install [Python3](https://www.python.org/downloads/)
2. Install [Git](https://git-scm.com/install/) on your platform
3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which manages the Python
   virtual environment and dependencies for this project. It installs its own Python 3.12, so no
   conda/pyenv setup is needed.
4. Clone this repository locally:

    ```bash
    git clone https://github.com/CursedRock17/matrix_lab_rover_above.git
    ```

### Repository Setup

1) From the project root, sync the environment. Pick the extra that matches your machine — this is
   the *only* difference between a GPU and a CPU-only setup:
    ```bash
    uv sync --extra cu121   # machine with an NVIDIA GPU
    uv sync --extra cpu     # no NVIDIA GPU (saves ~6 GB)
    ```

This reads `pyproject.toml`/`uv.lock` and builds a single `.venv/` on Python 3.12 containing every
package in the repo. It is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
the high-level folders (`ArUco_detector`, `YOLO_agent`, `LLM_hybrid`, `depth_anything_server`,
`rover_control`) *and* the simulator (`rover_mujoco`) are all installed as importable packages from
that one environment — no `PYTHONPATH`, no second env to activate, and a policy trained in sim is
importable from the real-rover code without leaving the venv.

2) Run anything in the repo through `uv run`, which uses that environment automatically:
    ```bash
    uv run rover_control/examples/aruco_pose_movement.py
    uv run rover_mujoco/scripts/teleop_rover.py
    ```

3) Before committing, lint, format, and type-check everything in one pass:
    ```bash
    uv run scripts/check.py         # report problems, change nothing
    uv run scripts/check.py --fix   # apply the formatter and ruff's safe autofixes
    ```

   The rules live in the root `pyproject.toml` under `[tool.ruff]` and follow the
   [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), at 100 columns
   rather than 80. `ruff` and `ty` come from the workspace's dev group, so a plain `uv sync`
   already installed them, and the same command covers `rover_mujoco/` too.

4) (Only for the `LLM_hybrid` examples) Install and start the local Ollama server — see [LLM_hybrid/README.md](LLM_hybrid/README.md) for the two-command setup.

---

## Project Structure

```
matrix_lab_rover_above/
├── ArUco_detector/          OpenCV-based ArUco marker detector. Reads a JPEG still from the
│                            ESP32 camera, finds any markers from a chosen dictionary, and
│                            returns each tag's (X, Y, Z) pose in meters.
├── YOLO_agent/              YOLOv8n object detector for COCO classes, plus a monocular depth
│                            estimator that pairs YOLO bounding boxes with Depth Anything V3
│                            to produce real-world (X, Y, Z) positions with no tag required.
├── LLM_hybrid/              Ollama integration for local vision LLMs. Can describe what the
│                            camera sees and output rover velocity commands as structured JSON.
├── depth_anything_server/   FastAPI server that offloads Depth Anything V3 inference to a
│                            shared desktop GPU. Rover laptops send a JPEG frame over HTTP and
│                            receive a full float32 depth map in return.
├── rover_control/           Rover base class and all runnable examples. Sends motor commands
│       └── examples/        over UDP at 10 Hz and reads encoder counts over the same link.
│                            All other folders are imported here — this is where students run code.
└── rover_mujoco/            MuJoCo simulation of the same differential-drive rover, for
        ├── envs/            developing control and RL policies before touching hardware.
        ├── scripts/         Ships teleop, a classical PID/bang-bang line follower, and PPO
        └── assets/          training for camera-only line following on generated tracks.
```

`rover_mujoco/` is a workspace member with its own `pyproject.toml`, but it shares the root
`uv.lock` and `.venv/` — the single `uv sync` above covers it. See
[rover_mujoco/README.md](rover_mujoco/README.md) for the simulation-specific docs.

---

## Examples

| Example | Folder | What it does |
|---------|--------|-------------|
| ArUco Pose Movement | [rover_control/examples/](rover_control/examples/) | Finds an ArUco tag, centers on it, and drives to ~0.25 m away. |
| ArUco Maze Runner | [rover_control/examples/](rover_control/examples/) | Navigates a sequence of ArUco tags in order using a continuous PID approach. |
| ArUco Maze Runner (Trapezoid) | [rover_control/examples/](rover_control/examples/) | Same maze, but uses a planned trapezoidal speed curve — more reliable in poor lighting. |
| ArUco Tag Tracker | [rover_control/examples/](rover_control/examples/) | Keeps a moving ArUco tag in frame at a fixed standoff distance, spinning to find it if lost. |
| YOLO Object Detection | [YOLO_agent/](YOLO_agent/) | Runs YOLOv8n on the laptop camera or ESP32 camera and returns detection dictionaries. |
| YOLO 3D Pose | [YOLO_agent/](YOLO_agent/) | Pairs YOLO detections with a depth model to give each object a real-world (X, Y, Z) position. |
| YOLO Object Navigation | [rover_control/examples/](rover_control/examples/) | Drives the rover toward a named COCO object using its YOLO 3D pose. |
| Depth Server | [depth_anything_server/](depth_anything_server/) | Offloads Depth Anything V3 inference to a desktop GPU; laptops send a JPEG and get a depth map back. |
| LLM Object Identification | [LLM_hybrid/](LLM_hybrid/) | Asks a local vision LLM what object is in frame and returns a structured JSON answer. |
| LLM Rover Driving | [LLM_hybrid/](LLM_hybrid/) | Proof of concept where the LLM watches the camera stream and outputs rover velocity commands. |
| Sim Manual Control | [rover_mujoco/scripts/](rover_mujoco/scripts/) | Drives the simulated rover around the MuJoCo viewer with WASD/arrow keys. |
| Sim Line Follower (PID) | [rover_mujoco/scripts/](rover_mujoco/scripts/) | Classical camera-only line following — PID or bang-bang — around a generated track. |
| Sim Line Follower (RL) | [rover_mujoco/scripts/](rover_mujoco/scripts/) | Trains and evaluates a PPO line-following policy on `LineFollower-v0` / `LineFollowerReal-v0`. |
| Sim Goal Navigation (RL) | [rover_mujoco/scripts/](rover_mujoco/scripts/) | Trains a PPO policy to drive to a commanded (x, y) offset in a 5x5 m arena while avoiding obstacles, using encoder odometry and the rover's lidar. |
| RL Line Follower (Real) | [rover_control/examples/](rover_control/examples/) | Runs the PPO policy trained in `rover_mujoco/` on the physical rover. |
