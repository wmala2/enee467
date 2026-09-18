# Lab Rover
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
    git clone https://github.com/wmala2/enee467.git
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

