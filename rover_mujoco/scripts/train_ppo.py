"""Train the PID-matched Gymnasium line follower with SB3 PPO, TensorBoard, and W&B."""

import argparse
from collections.abc import Callable
from functools import partial
import json
from pathlib import Path
import time
from typing import Any

import envs  # noqa: F401
from envs.line_dynamics import validate_ranges
from envs.tasks.line_follower_ppo_env import TRACKS
from evaluate_ppo import evaluate
from evaluate_ppo import write_metadata
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

ENV_ID = "LineFollowerPPO-v0"


def make_environment(track, reward_centering, dynamics="nominal", dr_ranges=None) -> gym.Env:
    # Each worker has a fixed track and samples start positions along it at reset.
    return Monitor(
        gym.make(
            ENV_ID,
            track=track,
            random_start=True,
            reward_centering=reward_centering,
            dynamics=dynamics,
            dr_ranges=dr_ranges,
        )
    )


class CompletionCallback(BaseCallback):
    """Select checkpoints using the weakest track's completion rate, then mean return."""

    def __init__(
        self,
        output,
        interval,
        episodes,
        reward_centering,
        tracks,
        dynamics="nominal",
        dr_ranges=None,
    ):
        super().__init__()
        self.output = output
        self.interval = interval
        self.episodes = episodes
        self.reward_centering = reward_centering
        self.tracks = tracks
        self.dynamics = dynamics
        self.dr_ranges = dr_ranges
        self.next_evaluation = interval
        self.best_score = (-1.0, -float("inf"))
        self.history = []
        self.outcomes = []

    def _on_training_start(self):
        # Include the incoming policy so fine-tuning cannot discard a stronger starting checkpoint.
        self._evaluate()

    def _evaluate(self):
        results = evaluate(
            self.model,
            episodes=self.episodes,
            seed=1000,
            reward_centering=self.reward_centering,
            tracks=self.tracks,
            dynamics=self.dynamics,
            dr_ranges=self.dr_ranges,
        )
        for track, result in results.items():
            for key in ("success_rate", "mean_return", "mean_deviation_cm", "mean_progress"):
                self.logger.record(f"eval/{track}/{key}", result[key])
        score = (
            min(result["success_rate"] for result in results.values()),
            float(np.mean([result["mean_return"] for result in results.values()])),
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.output / "best_model")
        self.history.append({"timesteps": self.num_timesteps, "tracks": results})
        (self.output / "validation.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        self.logger.dump(self.num_timesteps)
        print(
            f"Validation at {self.num_timesteps}: "
            + ", ".join(
                f"{track}={result['success_rate']:.0%}" for track, result in results.items()
            ),
            flush=True,
        )

    def _on_step(self):
        # Report episode-level physical metrics beside SB3's optimization diagnostics.
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.outcomes.append(info)
        if self.num_timesteps >= self.next_evaluation:
            self._evaluate()
            self.next_evaluation += self.interval
        return True

    def _on_rollout_end(self):
        if self.outcomes:
            for key in ("is_success", "mean_deviation_cm", "line_loss_frames", "progress_fraction"):
                self.logger.record(f"task/{key}", float(np.mean([o[key] for o in self.outcomes])))
            # Separate tracks so easy laps cannot hide poor learning on the figure eight.
            for track in self.tracks:
                outcomes = [outcome for outcome in self.outcomes if outcome["track"] == track]
                if outcomes:
                    for key in ("is_success", "mean_deviation_cm", "progress_fraction"):
                        self.logger.record(
                            f"task/{track}/{key}", float(np.mean([o[key] for o in outcomes]))
                        )
            self.outcomes.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tracks", nargs="+", choices=list(TRACKS), default=list(TRACKS))
    parser.add_argument(
        "--reward-centering", choices=("camera", "chassis", "both"), default="camera"
    )
    parser.add_argument("--dynamics", choices=("nominal", "bam", "dr"), default="nominal")
    parser.add_argument("--dr-ranges", type=Path, help="JSON parameter min/max overrides for DR")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("runs") / time.strftime("ppo-%Y%m%d-%H%M%S")
    )
    args = parser.parse_args()
    if (
        min(args.timesteps, args.n_envs, args.eval_every, args.eval_episodes, args.test_episodes)
        < 1
    ):
        parser.error("step, worker, and episode counts must be positive")
    if args.n_envs < len(args.tracks):
        parser.error(
            f"use at least {len(args.tracks)} workers so every selected track participates"
        )
    if not 0 < args.gamma <= 1:
        parser.error("gamma must be in (0, 1]")
    if args.dr_ranges and args.dynamics != "dr":
        parser.error("--dr-ranges requires --dynamics dr")
    # Store resolved bounds so evaluation does not depend on a later edit to the range file.
    overrides = json.loads(args.dr_ranges.read_text()) if args.dr_ranges else None
    dr_ranges = validate_ranges(overrides) if args.dynamics == "dr" else None
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hyperparameters: dict[str, Any] = {
        "learning_rate": args.learning_rate,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": args.gamma,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": args.ent_coef,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [64, 64], "log_std_init": -1.0},
    }
    config = {
        **vars(args),
        "output": str(args.output),
        "resume": str(args.resume) if args.resume else None,
        "env_id": ENV_ID,
        "tracks": args.tracks,
        "domain_randomization": args.dynamics == "dr",
        "dr_ranges": dr_ranges,
        "actuator_model": "velocity_servo" if args.dynamics == "nominal" else "bam_stribeck_dc",
        "control_hz": 10,
        "tape_width_m": 0.0508,
        "camera_tilt_deg": 45,
        "camera_fovy_deg": 60,
        "max_deviation_m": 0.06,
        "finish_tolerance_m": 0.03,
        "hyperparameters": hyperparameters,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run = None
    environment = None
    try:
        # W&B is enabled by default and must initialize successfully before training starts.
        if not args.no_wandb:
            import wandb

            run = wandb.init(
                project="rover-line-follower",
                name=args.output.name,
                group=args.wandb_group,
                config=config,
                sync_tensorboard=True,
                dir=str(args.output),
            )
            print(f"W&B: {run.url}", flush=True)
        factories: list[Callable[[], gym.Env]] = [
            partial(
                make_environment,
                args.tracks[i % len(args.tracks)],
                args.reward_centering,
                args.dynamics,
                dr_ranges,
            )
            for i in range(args.n_envs)
        ]
        environment = SubprocVecEnv(factories, start_method="spawn")
        environment.seed(args.seed)
        # Resumed policies use the same explicit hyperparameters recorded for this run.
        if args.resume:
            model = PPO.load(
                args.resume,
                env=environment,
                device="cpu",
                seed=args.seed,
                tensorboard_log=str(args.output / "tensorboard"),
                **hyperparameters,
            )
        else:
            model = PPO(
                "MlpPolicy",
                environment,
                device="cpu",
                seed=args.seed,
                verbose=0,
                tensorboard_log=str(args.output / "tensorboard"),
                **hyperparameters,
            )
        # Carry the source run's lineage because SB3 resets its counter on every continuation.
        prior_timesteps = model.num_timesteps
        if args.resume:
            source_directory = args.resume.parent
            if source_directory.name == "checkpoints":
                source_directory = source_directory.parent
            source_config = source_directory / "config.json"
            if source_config.exists():
                prior_timesteps += json.loads(source_config.read_text(encoding="utf-8")).get(
                    "prior_timesteps", 0
                )
        config["checkpoint_timesteps"] = model.num_timesteps
        config["prior_timesteps"] = prior_timesteps
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if run is not None:
            run.config.update({
                "checkpoint_timesteps": model.num_timesteps,
                "prior_timesteps": prior_timesteps,
            })
        callbacks: list[BaseCallback] = [
            CompletionCallback(
                args.output,
                args.eval_every,
                args.eval_episodes,
                args.reward_centering,
                args.tracks,
                args.dynamics,
                dr_ranges,
            ),
            CheckpointCallback(
                save_freq=max(1, args.eval_every // args.n_envs),
                save_path=str(args.output / "checkpoints"),
            ),
        ]
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        model.save(args.output / "final_model")
        if not (args.output / "best_model.zip").exists():
            model.save(args.output / "best_model")
        environment.close()
        environment = None
        # Final evaluation uses seeds excluded from checkpoint selection and records each episode.
        best = PPO.load(args.output / "best_model", device="cpu")
        evaluation_dir = args.output / "evaluation"
        evaluation_dir.mkdir()
        results = evaluate(
            best,
            args.test_episodes,
            10000,
            evaluation_dir,
            video=True,
            reward_centering=args.reward_centering,
            tracks=args.tracks,
            dynamics=args.dynamics,
            dr_ranges=dr_ranges,
        )
        write_metadata(
            args.output / "best_model.zip",
            evaluation_dir,
            results,
            args.dynamics,
            dr_ranges,
            args.reward_centering,
            10000,
            args.test_episodes,
        )
        print(
            "Final evaluation: "
            + ", ".join(
                f"{track}={result['success_rate']:.0%}" for track, result in results.items()
            ),
            flush=True,
        )
        if run is not None:
            import wandb

            for track, result in results.items():
                run.summary[f"test/{track}/success_rate"] = result["success_rate"]
                run.summary[f"test/{track}/mean_deviation_cm"] = result["mean_deviation_cm"]
            run.log({"test/trajectories": wandb.Image(str(evaluation_dir / "trajectories.png"))})
            artifact = wandb.Artifact(args.output.name, type="model")
            artifact.add_file(str(args.output / "best_model.zip"))
            artifact.add_file(str(args.output / "config.json"))
            artifact.add_file(str(evaluation_dir / "evaluation.json"))
            run.log_artifact(artifact)
        print(f"Saved model and evaluation: {args.output}", flush=True)
    finally:
        if environment is not None:
            environment.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
