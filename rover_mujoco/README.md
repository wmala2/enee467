# Rover MuJoCo
This is a MuJoCo environment which will fascilitate the simulation and training of our differential drive, Mini Claw STEM rovers in a easy to setup, fast, simple, yet robust training environments

## Prerequisites
This environment should be system-agnostic since we're using dedicated Python virtual environments without (for now) requiring access to a GPU, so it should work on Linux, Mac, and Windows.
Prior to starting, we recommend installing the following software:
- **Python 3.12** (see `requires-python` in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, which manages the Python virtual environment and dependencies
- **[VSCode](https://code.visualstudio.com/download)** (optional, but recommended): this repo ships a `.vscode/extensions.json` with recommended extensions (Python, Ruff, a MuJoCo viewer)

It's not a requirement, but you should be familiar with the [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/overview.html) and it's useful to know basic [Reinforcement Learning Concepts](https://huggingface.co/learn/deep-rl-course/en/unit1/rl-framework)

## Getting Started
All of the development for creating scripts will occur in Python, thus we'll be using [uv](https://docs.astral.sh/uv/) to maintain a Python virtual environment.

Clone the repo and install the provided dependencies:

```shell
cd ~/Documents
git clone https://github.com/CursedRock17/rover_mujoco.git -b main
cd rover_mujoco
uv sync
```

That's it: `uv sync` reads `pyproject.toml`/`uv.lock` and creates a `.venv` with everything
this project needs (no separate `uv init` step: the repo already has its own `pyproject.toml`
from the clone).

See [`docs/workspace-setup.md`](docs/workspace-setup.md) for the full walkthrough: verifying
MuJoCo actually opens a window, adding your own packages on top of the baseline set, and notes
for Windows/Mac/Linux.

## Exploring
Now that you've got the repository setup, we recommend starting with the [manual control](scripts/teleop_rover.py) example to see that the 3D MuJoCo viewer works and the rover implementation is clean:
In a `uv sync`'d shell, run:

```shell
cd ~/Documents/rover_mujoco
uv run scripts/teleop_rover.py
```
You should be able to use WASD/Arrow Key controls to drive a rover around following differential drive characteristics.

**Going Further**
The main goal of this directory is simulation to real life transfer with our Mini Claw STEM Rover which will allow individuals using the rovers to test their algorithms, control, RL policies, and more in simulation then deploy them in real life.
The following table denotes the out of box capabilites of the current environment

| Task | Descriptions |
|------|--------------|
| Manual Control | Use WASD/Arrow Keys to manually drive around the rover in a realistic manner following differential drive kinematics |

