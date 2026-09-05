"""Push the trained LineFollowerReal-v0 PPO policy to Hugging Face Hub, along with a model
card documenting the environment, reward function, hyperparameters, and known sim-to-real
caveats — not just the raw weights, so anyone (including us, later) can reproduce or extend
this run without having to reconstruct the setup from scratch.

Usage: uv run python scripts/push_to_hub.py <repo_id> [--private/--public]
Requires: `hf auth login` already done (checked via huggingface_hub.whoami()).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi
from huggingface_hub import whoami
from train_real import ENV_ID
from train_real import MODEL_DIR
from train_real import N_ENVS

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCE_FILES = [
    "envs/tasks/line_follower_real_env.py",
    "envs/tasks/line_follower_env.py",
    "envs/motor.py",
    "envs/tracks.py",
    "scripts/train_real.py",
    "docs/rl-line-follower.md",
]


def build_model_card(training_log_summary: str) -> str:
    return f"""---
tags:
- reinforcement-learning
- stable-baselines3
- mujoco
- ppo
- line-following
- robotics
library_name: stable-baselines3
---

# Line-Following PPO Policy (`{ENV_ID}`)

PPO policy trained in MuJoCo to drive a differential-drive rover ("Mini Claw STEM Rover")
along a black line track, for eventual deployment on the real rover via a laptop bridge.
Trained in [rover_mujoco](https://github.com/CursedRock17/rover_mujoco).

## Observation space (`Dict`)
- `image`: 64x64 grayscale onboard camera frame, refreshed at 10 Hz (matches the real
  rover's command rate), under domain randomization (white balance, brightness, pixel
  noise, effective resolution).
- `encoders`: left/right wheel encoder **ticks accrued since the last control step**
  (quantized, `ENCODER_CPR_WHEEL=680`, matching the real firmware's raw quadrature-count
  query) — not velocity; the real rover has no velocity sensor.
- No ground-truth position anywhere in the observation.

## Action space
`Box(-10, 10, shape=(2,))`: left/right wheel angular velocity target, rad/s. **Known,
unresolved gaps before real deployment**:
- The real firmware's `"m"` UDP command takes `left_mps`/`right_mps` (m/s), not rad/s —
  needs a wheel-radius conversion in the deployment bridge.
- Sim's "same sign on both wheels" spins in place; the real rover's convention is same
  sign = forward (confirmed from `wmala2/rover-firmware`). The deployment bridge must
  negate one channel.

## Reward
`PROGRESS_WEIGHT * (forward arc-length delta / track length) + CENTER_WEIGHT * (1 - |line-centering error|) + COMPLETION_BONUS` on
lap/traversal completion. Progress is ground-truth position projected onto the track's
waypoints (reward-only privilege — never in the observation), windowed around the
previous match to avoid teleporting to a spatially-close-but-path-distant part of the
track. See `envs/tasks/line_follower_env.py` for the full reasoning and the specific
failure modes (a "vibrate in place" exploit, and a reward-spike bug) this design fixes.

## Training setup
- Algorithm: PPO (`MultiInputPolicy`, stable-baselines3). SB3 defaults for everything
  **except** `ent_coef=0.01` (SB3 default is 0.0) — a hyperparameter-sweep result: an
  initial run with default hyperparameters (`scripts/train_real.py`, 10M steps) plateaued
  around ep_rew_mean~90-105 with zero laps/finishes and perfectly reproducible (deterministic)
  rollouts, indicating premature convergence to an overly-cautious policy. A small sweep
  (`scripts/sweep_hparams.py`) found `ent_coef=0.01` broke past that plateau immediately (3M
  steps already exceeded the baseline's 10M-step peak reward), so training continued from
  that checkpoint (`scripts/continue_sweep_winner.py`) to 10M total steps.
- n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4 (constant), clip_range=0.2.
- `{N_ENVS}` parallel envs (`SubprocVecEnv`, `fork` start method), GPU (`device="auto"`).
- Total timesteps: 10,000,000 (summed across all parallel envs; 3M initial + 7M continued).
- Control-loop rate: 10 Hz (`CONTROL_HZ`), decimated from a 500 Hz physics integrator.
- Domain randomization: action noise/latency, wheel-floor friction, encoder noise, camera
  FOV/brightness/white-balance/pixel-noise/effective-resolution — see
  `docs/rl-line-follower.md`'s DR table for exact ranges and reasoning.

## Training run summary
{training_log_summary}

## Files in this repo
- `model.zip` — the trained SB3 policy (`PPO.load("model.zip")`)
- `source/` — snapshot of the env/reward/motor-model/training-script source at the time
  this policy was trained, for exact reproducibility
- This README

## Caveats
Never validated on real hardware yet. Domain randomization narrows the sim-to-real gap;
it doesn't close it. See `docs/rl-line-follower.md` in the source snapshot for the full
list of open items (action format/sign convention, camera hardware location still
unconfirmed, motor model is datasheet-derived not bench-measured).
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="e.g. CursedRock17/rover-line-follower-ppo")
    parser.add_argument(
        "--public", action="store_true", help="Make the repo public (default: private)"
    )
    parser.add_argument(
        "--summary", default="(fill in final reward/laps/displacement stats before pushing)"
    )
    args = parser.parse_args()

    who = whoami()
    print(f"Authenticated as {who['name']}")

    model_path = os.path.join(MODEL_DIR, "model.zip")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path} — train first.", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    api.create_repo(args.repo_id, private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(model_path, os.path.join(tmp, "model.zip"))
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(build_model_card(args.summary))

        src_dir = os.path.join(tmp, "source")
        for rel in SOURCE_FILES:
            dst = os.path.join(src_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(os.path.join(REPO_ROOT, rel), dst)

        with open(os.path.join(tmp, "hyperparameters.json"), "w") as f:
            json.dump(
                {
                    "env_id": ENV_ID,
                    "total_timesteps": 10_000_000,
                    "n_envs": N_ENVS,
                    "algorithm": "PPO",
                    "policy": "MultiInputPolicy",
                    "ent_coef": 0.01,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "learning_rate": 3e-4,
                    "clip_range": 0.2,
                    "provenance": "3M steps at ent_coef=0.01 (scripts/sweep_hparams.py) "
                    "+ 7M continued (scripts/continue_sweep_winner.py)",
                },
                f,
                indent=2,
            )

        api.upload_folder(folder_path=tmp, repo_id=args.repo_id)

    print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
