"""Hyperparameter sweep for LineFollowerReal-v0, scored per track.

    uv run python scripts/sweep_line_follower.py                 # run every variant
    uv run python scripts/sweep_line_follower.py --variant wide  # just one
    uv run python scripts/sweep_line_follower.py --list

Each variant trains, then evaluates on *each track separately*. That separation is the point:
the tracks are not equally hard, and an aggregate score hides it. With the track pinned, the
current baseline is 20/20 completions on the oval and 19/20 on the s-curve, so a sweep that
reports one blended number cannot tell you whether a variant fixed the s-curve or just drew
the oval more often.

Every run writes TensorBoard, and --wandb adds Weights & Biases. With a `wandb login` on the
machine the runs stream live; without one they fall back to offline mode, which needs no account
and uploads nothing, and `wandb sync wandb/` pushes them later if you change your mind.
"""

import argparse
import os
import subprocess
import sys

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
    from envs.tasks.line_follower_real_env import EVAL_TRACKS_REAL
    from envs.tasks.line_follower_real_env import TRACKS_REAL

    results = {}
    # Training tracks first, then the held-out ones. The held-out numbers are the ones that
    # answer "will this survive a track it has not seen", which is the deployment question;
    # the training numbers are only there to show whether a gap opened up.
    for track in list(TRACKS_REAL) + list(EVAL_TRACKS_REAL):
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
        # Deviation only over episodes that finished the track. Unconditioned it is worse than
        # useless: a rover driving dead straight scores 3.4 cm on the oval -- better than a
        # trained policy and near the classical baseline -- because it loses the line early and
        # the average only ever sees the first metre, where every controller is still centred.
        finished = [o for o in outcomes if o["completed"]]
        results[track] = {
            "completion": sum(o["completed"] for o in outcomes) / episodes,
            "mean_dev_cm": float(np.mean([o["mean_dev"] for o in finished]) * 100)
            if finished
            else float("nan"),
            "worst_dev_cm": float(np.max([o["max_dev"] for o in finished]) * 100)
            if finished
            else float("nan"),
        }
    return results


def print_scores(scores):
    from envs.tasks.line_follower_real_env import EVAL_TRACKS_REAL

    for track, s in scores.items():
        print(
            f"  {track:32s} {'held-out' if track in EVAL_TRACKS_REAL else 'training':9s} "
            f"completion {s['completion']:6.1%}  "
            f"mean {s['mean_dev_cm']:5.1f} cm  worst {s['worst_dev_cm']:6.1f} cm"
        )


def _wandb_has_credentials():
    return bool(os.environ.get("WANDB_API_KEY")) or "api.wandb.ai" in _netrc_text()


def _netrc_text():
    try:
        with open(os.path.expanduser("~/.netrc"), encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


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
    print_scores(scores)
    if run is not None:
        run.log({f"eval/{t}/{k}": v for t, s in scores.items() for k, v in s.items()})
        run.finish()
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", help="comma-separated variants to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list variants and exit")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="re-score already-trained variants without retraining them",
    )
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

    if args.wandb and not _wandb_has_credentials():
        # No login, so log locally rather than failing the run: nothing leaves the machine, and
        # `wandb sync wandb/` uploads it afterwards if wanted.
        os.environ.setdefault("WANDB_MODE", "offline")
        print("wandb: no credentials found, logging offline to ./wandb/")

    names = args.variant.split(",") if args.variant else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    # One fresh process per variant. Training forks its workers with SubprocVecEnv, and a GL
    # context does not survive fork: once this process has built a renderer -- which scoring a
    # variant does -- every forked worker of the *next* variant dies the moment it constructs
    # its own, showing up as ConnectionResetError from a worker that never sent its spaces.
    # Renderer.close() is not enough; the parent has to have never touched GL. Re-exec instead.
    if len(names) > 1 and not args.score_only:
        for name in names:
            argv = [sys.executable, __file__, "--variant", name, "--episodes", str(args.episodes)]
            if args.wandb:
                argv.append("--wandb")
            if args.timesteps:
                argv += ["--timesteps", str(args.timesteps)]
            result = subprocess.run(argv, check=False)
            if result.returncode != 0:
                print(f"{name}: exited {result.returncode}, continuing with the next variant")
        return

    summary = {}
    for name in names:
        if args.score_only:
            model_path = os.path.join(RUNS_DIR, name, "model")
            if not os.path.exists(model_path + ".zip"):
                print(f"{name}: no model at {model_path}.zip, skipping")
                continue
            print(f"\n=== {name} (scoring only)")
            summary[name] = evaluate_per_track(model_path, args.episodes)
            print_scores(summary[name])
        else:
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
