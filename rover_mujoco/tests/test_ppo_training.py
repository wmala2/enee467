"""Verify that resumed training honors its recorded experiment configuration."""

import json
from pathlib import Path
import subprocess
import sys

from envs.tasks.line_follower_ppo_env import LineFollowerPPOEnv
from envs.tasks.line_follower_ppo_env import TRACKS
import pytest
from stable_baselines3 import PPO
import torch


def test_resume_uses_requested_hyperparameters(tmp_path):
    # Save a real trained checkpoint with parameters different from the continuation request.
    torch.set_num_threads(1)
    environment = LineFollowerPPOEnv(max_steps=2)
    try:
        model = PPO(
            "MlpPolicy",
            environment,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            learning_rate=0.0003,
            ent_coef=0.005,
            policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.0},
            seed=7,
        )
        model.learn(16)
        checkpoint = tmp_path / "source.zip"
        model.save(checkpoint)
    finally:
        environment.close()

    # Include earlier interactions to exercise cumulative provenance across multiple resumptions.
    (tmp_path / "config.json").write_text(json.dumps({"prior_timesteps": 100}), encoding="utf-8")
    # Exercise the actual CLI, worker processes, checkpoint serialization, and final evaluation.
    output = tmp_path / "continued"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/train_ppo.py"),
            "--resume",
            str(checkpoint),
            "--output",
            str(output),
            "--timesteps",
            "1",
            "--n-envs",
            # Every track must get a worker, so this tracks the registry rather than a literal.
            str(len(TRACKS)),
            "--test-episodes",
            "1",
            "--learning-rate",
            "0.0001",
            "--ent-coef",
            "0.001",
            "--gamma",
            "0.98",
            "--no-wandb",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    resumed = PPO.load(output / "final_model.zip")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert (
        resumed.learning_rate == config["hyperparameters"]["learning_rate"] == pytest.approx(0.0001)
    )
    assert resumed.ent_coef == config["hyperparameters"]["ent_coef"] == pytest.approx(0.001)
    assert resumed.policy.optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    assert resumed.n_steps == config["hyperparameters"]["n_steps"] == 512
    assert resumed.gamma == config["hyperparameters"]["gamma"] == pytest.approx(0.98)
    assert config["checkpoint_timesteps"] == 16
    assert config["prior_timesteps"] == 116
    history = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert history[0]["timesteps"] == 0
    assert list(history[0]["tracks"]) == list(TRACKS)
    # SB3 finishes whole rollouts, so a 1-step request still collects n_steps per worker.
    assert resumed.num_timesteps == len(TRACKS) * resumed.n_steps
    assert (output / "evaluation/evaluation.json").is_file()
