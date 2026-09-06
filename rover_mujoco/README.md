# Rover MuJoCo
This is a MuJoCo environment which will fascilitate the simulation and training of our differential drive, Mini Claw STEM rovers in a easy to setup, fast, simple, yet robust training environments

## Prerequisites
This environment should be system-agnostic since we're using dedicated Python virtual environments without (for now) requiring access to a GPU, so it should work on Linux, Mac, and Windows.
Prior to starting, we recommend installing the following software:
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, which manages the Python virtual environment and dependencies. It installs Python 3.12 (see `requires-python` in `pyproject.toml`) for you.
- **[VSCode](https://code.visualstudio.com/download)** (optional, but recommended): this repo ships a `.vscode/extensions.json` with recommended extensions (Python, Ruff, a MuJoCo viewer)

It's not a requirement, but you should be familiar with the [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/overview.html) and it's useful to know basic [Reinforcement Learning Concepts](https://huggingface.co/learn/deep-rl-course/en/unit1/rl-framework)

## Getting Started
All of the development for creating scripts will occur in Python, thus we'll be using [uv](https://docs.astral.sh/uv/) to maintain a Python virtual environment.

This directory is a member of the parent repo's [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
it keeps its own `pyproject.toml`, but shares one `uv.lock` and one `.venv/` with the high-level
rover code. So clone and sync from the **repository root**, not from here:

```shell
cd ~/Documents
git clone https://github.com/CursedRock17/matrix_lab_rover_above.git
cd matrix_lab_rover_above
uv sync --extra cpu   # or --extra cu121 on an NVIDIA machine
```

That single sync installs MuJoCo, Gymnasium, and stable-baselines3 alongside the ArUco/YOLO/LLM
packages, so a policy trained here is importable from `rover_control/` without switching
environments. There's no separate `uv init` or a second `uv sync` in this folder.

See [`docs/workspace-setup.md`](docs/workspace-setup.md) for the full walkthrough: verifying
MuJoCo actually opens a window, adding your own packages on top of the baseline set, and notes
for Windows/Mac/Linux.

## Exploring
Now that you've got the repository setup, we recommend starting with the [manual control](scripts/teleop_rover.py) example to see that the 3D MuJoCo viewer works and the rover implementation is clean:
In a `uv sync`'d shell, run:

```shell
cd ~/Documents/matrix_lab_rover_above
uv run rover_mujoco/scripts/teleop_rover.py
```
You should be able to use WASD/Arrow Key controls to drive a rover around following differential drive characteristics.

**Going Further**
The main goal of this directory is simulation to real life transfer with our Mini Claw STEM Rover which will allow individuals using the rovers to test their algorithms, control, RL policies, and more in simulation then deploy them in real life.
The following table denotes the out of box capabilites of the current environment

| Task | Script | Descriptions |
|------|--------|--------------|
| Manual Control | [`scripts/teleop_rover.py`](scripts/teleop_rover.py) | Use WASD/Arrow Keys to manually drive around the rover in a realistic manner following differential drive kinematics ([docs](docs/simple-control.md)) |
| Track Generation | [`scripts/gen_track.py`](scripts/gen_track.py) | Generate a black line track (oval or s-curve) as an MJCF fragment from a list of (x, y) waypoints, for the line-following tasks below |
| Line Following (Classical) | [`scripts/line_follower.py`](scripts/line_follower.py) | Follow a track using only the onboard camera with a PID or bang-bang controller, with a tiltable camera mount — the classical-control baseline ([docs](docs/pid-line-follower.md)) |
| Line Following (RL) | [`scripts/train.py`](scripts/train.py), [`scripts/evaluate.py`](scripts/evaluate.py) | Train and evaluate a PPO policy on `LineFollower-v0`: camera-only observations with domain randomization ([docs](docs/rl-line-follower.md)) |
| Line Following (Sim-to-Real) | [`scripts/train_real.py`](scripts/train_real.py), [`scripts/evaluate_real.py`](scripts/evaluate_real.py) | Same task on `LineFollowerReal-v0`, constrained to what the physical rover actually has: camera + noisy wheel encoders, wheel-velocity actions, and randomized motor lag/latency ([docs](docs/rl-line-follower.md)) |
| Goal Navigation (RL) | [`scripts/train_goal_nav.py`](scripts/train_goal_nav.py), [`scripts/evaluate_goal_nav.py`](scripts/evaluate_goal_nav.py) | Drive to a commanded (x, y) offset — "drive to (1, 2)" — inside a walled 5x5 m arena while avoiding randomly placed obstacles, navigating by dead reckoning from the wheel encoders plus the rover's three-beam lidar ([docs](docs/rl-goal-nav.md)) |
| Arena Generation | [`scripts/gen_arena.py`](scripts/gen_arena.py) | Generate the walled 5x5 m arena and its pool of movable obstacles as an MJCF fragment, for the goal-navigation task above |
| Policy Sharing | [`scripts/push_to_hub.py`](scripts/push_to_hub.py) | Push a trained policy to the Hugging Face Hub with a model card covering the environment, reward, hyperparameters, and sim-to-real caveats |
| Hyperparameter Sweeps | [`scripts/sweep_hparams.py`](scripts/sweep_hparams.py), [`scripts/continue_sweep_winner.py`](scripts/continue_sweep_winner.py) | Run short PPO variants against the baseline to compare learning curves cheaply, then resume training from the best checkpoint |

Trained policies are deployed to the physical rover from the parent repo — see
`rover_control/rl_rover.py` and `rover_control/examples/rl_line_follower.py`.

