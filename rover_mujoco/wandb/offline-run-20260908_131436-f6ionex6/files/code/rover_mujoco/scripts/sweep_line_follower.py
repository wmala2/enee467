"""Hyperparameter sweep for LineFollowerReal-v0, scored per track.

    uv run python scripts/sweep_line_follower.py                 # run every variant
    uv run python scripts/sweep_line_follower.py --variant wide  # just one
    uv run python scripts/sweep_line_follower.py --list

Each variant trains, then evaluates on *each track separately*. That separation is the point:
the tracks are not equally hard, and an aggregate score hides it. With the track pinned, the
current baseline is 20/20 completions on the oval and 19/20 on the s-curve, so a sweep that
reports one blended number cannot tell you whether a variant fixed the s-curve or just drew
the oval more often.

Every run writes TensorBoard, and --wandb adds Weights & Biases. W&B is left in offline mode by
default (WANDB_MODE=offline) so it needs no account and uploads nothing; `wandb sync wandb/`
pushes the runs later if you want the cross-run comparison and sweep plots.
"""

import argparse
import os

import envs  # noqa: F401  (imported for its side effect: registers LineFollowerReal-v0)
from evaluate_real import episode_outcome
from evaluate_real import track_deviation
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from train_real import ENV_ID
from train_real import N_ENVS
from wandb_logging import start_run

RUNS_DIR = os.path.join(os.path.dirname(__file__), "../runs/sweep_line_follower")

# Variants chosen against what the evaluation actually says is wrong. The failure is rare,
# confined to the s-curve, and looks like losing the line on its tighter bends -- so these
# probe exploration, how far ahead the policy plans, and network capacity, rather than
# re-litigating the reward, which three iterations already showed is not the constraint.
VARIANTS = {
    "baseline": {},
    "longer": {"total_timesteps": 900_000},
    "explore": {"ent_coef": 0.02},
    "farsighted": {"gamma": 0.999},
    "bigger_net": {"policy_kwargs": {"net_arch": [128, 128]}},
    "big_batch": {"n_steps": 1024, "batch_size": 1024},
}

DEFAULTS = {
    "total_timesteps": 300_000,
    "n_steps": 512,
    "batch_size": 512,
    "gamma": 0.995,
    "ent_coef": 0.01,
}


def evaluate_per_track(model_path, episodes=20):
    """Completion rate and deviation on each track separately, reusing evaluate_real.py's
    metrics so a sweep number and a hand-run evaluation always mean the same thing."""
    from envs.tasks.line_follower_real_env import TRACKS_REAL

    results = {}
    for track in TRACKS_REAL:
        env = gym.make(ENV_ID, track=track)
        model = PPO.load(model_path, env=env)
        outcomes = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=episode)
            devs = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                devs.append(track_deviation(env, env.unwrapped.data.qpos[:2].copy()))
                if terminated or truncated:
                    break
            out = episode_outcome(env)
            out["mean_dev"] = float(np.mean(devs))
            out["max_dev"] = float(np.max(devs))
            outcomes.append(out)
        env.close()
        results[track] = {
            "completion": sum(o["completed"] for o in outcomes) / episodes,
            "mean_dev_cm": float(np.mean([o["mean_dev"] for o in outcomes]) * 100),
            "worst_dev_cm": float(np.max([o["max_dev"] for o in outcomes]) * 100),
        }
    return results


def run_variant(name, use_wandb, timesteps=None, episodes=20):
    config = {**DEFAULTS, **VARIANTS[name]}
    if timesteps is not None:
        config["total_timesteps"] = timesteps
    model_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n=== {name}: {config}")

    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    run, callback = (
        start_run("rover-line-follower-sweep", name, config, model_dir)
        if use_wandb
        else (None, None)
    )
    budget = config.pop("total_timesteps")
    model = PPO("MultiInputPolicy", env, verbose=0, seed=0, tensorboard_log=model_dir, **config)
    model.learn(total_timesteps=budget, callback=callback)
    model.save(os.path.join(model_dir, "model"))
    env.close()

    scores = evaluate_per_track(os.path.join(model_dir, "model"), episodes)
    for track, s in scores.items():
        print(
            f"  {track:28s} completion {s['completion']:6.1%}  "
            f"mean {s['mean_dev_cm']:5.1f} cm  worst {s['worst_dev_cm']:6.1f} cm"
        )
    if run is not None:
        run.log({f"eval/{t}/{k}": v for t, s in scores.items() for k, v in s.items()})
        run.finish()
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", help="run only this variant")
    parser.add_argument("--list", action="store_true", help="list variants and exit")
    parser.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    parser.add_argument(
        "--timesteps", type=int, help="override every variant's budget (smoke test)"
    )
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes per track")
    args = parser.parse_args()

    if args.list:
        for name, overrides in VARIANTS.items():
            print(f"  {name:12s} {overrides or 'defaults'}")
        return

    if args.wandb:
        # Offline unless the caller has explicitly chosen otherwise: no account needed and
        # nothing leaves the machine. `wandb sync wandb/` uploads afterwards.
        os.environ.setdefault("WANDB_MODE", "offline")

    names = [args.variant] if args.variant else list(VARIANTS)
    summary = {}
    for name in names:
        summary[name] = run_variant(name, args.wandb, args.timesteps, args.episodes)

    print("\n=== sweep summary (completion per track) ===")
    for name, scores in summary.items():
        cells = "  ".join(
            f"{t.replace('rover_line_', '').replace('_real.xml', '')}: "
            f"{s['completion']:.0%}/{s['mean_dev_cm']:.1f}cm"
            for t, s in scores.items()
        )
        print(f"  {name:12s} {cells}")


if __name__ == "__main__":
    main()
