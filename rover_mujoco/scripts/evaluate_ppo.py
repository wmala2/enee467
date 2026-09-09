"""Evaluate a saved PPO policy with independent lap-completion metrics and camera video."""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import cast

import envs  # noqa: F401
from envs.tasks.line_follower_ppo_env import LineFollowerPPOEnv
from envs.tasks.line_follower_ppo_env import TRACKS
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch


def evaluate(
    model,
    episodes=20,
    seed=10000,
    output=None,
    human=False,
    video=False,
    reward_centering="camera",
):
    """Use fixed starts with fresh perturbation seeds and deterministic policy actions."""
    summary = {}
    trajectories = {}
    for track in TRACKS:
        environment = gym.make(
            "LineFollowerPPO-v0",
            track=track,
            render_mode="human" if human else None,
            reward_centering=reward_centering,
        )
        outcomes, paths = [], []
        writer = None
        try:
            for episode in range(episodes):
                observation, _ = environment.reset(seed=seed + episode)
                rows, total_reward = [], 0.0
                if video and episode == 0 and output is not None:
                    import cv2

                    writer = cv2.VideoWriter(
                        str(output / f"{track}_camera.mp4"),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        10,
                        (384, 384),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create the evaluation video")
                while True:
                    start = time.monotonic()
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = environment.step(action)
                    total_reward += float(reward)
                    base = cast(LineFollowerPPOEnv, environment.unwrapped)
                    rows.append({
                        "time_s": base.steps * base.dt,
                        "x_m": float(base.data.qpos[0]),
                        "y_m": float(base.data.qpos[1]),
                        "deviation_cm": info["deviation_cm"],
                        "progress_fraction": info["progress_fraction"],
                        "near_error": float(observation[0]),
                        "near_visible": bool(observation[2]),
                        "forward_action": float(action[0]),
                        "steering_action": float(action[1]),
                    })
                    if writer is not None:
                        import cv2

                        writer.write(
                            cv2.resize(
                                base.frame[:, :, ::-1], (384, 384), interpolation=cv2.INTER_NEAREST
                            )
                        )
                    if human:
                        time.sleep(max(0, base.dt - (time.monotonic() - start)))
                    if terminated or truncated:
                        break
                if writer is not None:
                    writer.release()
                    writer = None
                outcomes.append({**info, "seed": seed + episode, "return": total_reward})
                paths.append(rows)
                if output is not None:
                    with (output / f"{track}_{seed + episode}.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as file:
                        csv_writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                        csv_writer.writeheader()
                        csv_writer.writerows(rows)
            summary[track] = {
                "success_rate": float(np.mean([o["is_success"] for o in outcomes])),
                "mean_return": float(np.mean([o["return"] for o in outcomes])),
                "mean_deviation_cm": float(np.mean([o["mean_deviation_cm"] for o in outcomes])),
                "max_deviation_cm": float(max(o["max_deviation_cm"] for o in outcomes)),
                "mean_progress": float(np.mean([o["progress_fraction"] for o in outcomes])),
                "episodes": outcomes,
            }
            trajectories[track] = paths
        finally:
            if writer is not None:
                writer.release()
            environment.close()
    if output is not None:
        (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        plot_trajectories(trajectories, summary, output)
    return summary


def plot_trajectories(trajectories, summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot every episode, including failures, against the exact waypoint centerline.
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (track, paths) in zip(axes, trajectories.items()):
        points = np.asarray(TRACKS[track]())
        axis.plot(points[:, 0], points[:, 1], "k--", label="track center")
        for rows in paths:
            axis.plot([r["x_m"] for r in rows], [r["y_m"] for r in rows], alpha=0.5)
        axis.set(
            title=f"{track}: {summary[track]['success_rate']:.0%} completed",
            xlabel="x (m)",
            ylabel="y (m)",
        )
        axis.set_aspect("equal")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "trajectories.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--reward-centering", choices=("camera", "chassis", "both"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    output = args.output or args.model.parent / f"evaluation-{int(time.time())}"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    # Recover reward provenance for older checkpoints as well as newly trained policies.
    config_path = args.model.parent / "config.json"
    if args.model.parent.name == "checkpoints":
        config_path = args.model.parent.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    centering = args.reward_centering or config.get("reward_centering", "camera")
    summary = evaluate(
        model,
        args.episodes,
        args.seed,
        output,
        args.viewer,
        args.video,
        reward_centering=centering,
    )
    for track, result in summary.items():
        print(
            f"{track}: {result['success_rate']:.0%} completed, "
            f"mean error {result['mean_deviation_cm']:.2f} cm"
        )
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
