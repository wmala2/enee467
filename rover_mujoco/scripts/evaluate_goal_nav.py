"""Evaluates a trained GoalNav-v0 policy, either in the viewer or as a batch of headless
episodes that reports how often it actually arrives.

    uv run python scripts/evaluate_goal_nav.py                  # viewer, random goals
    uv run python scripts/evaluate_goal_nav.py 1 2              # viewer, "drive to (1, 2)"
    uv run python scripts/evaluate_goal_nav.py --episodes 200   # headless, success statistics

Evaluation always runs the *full* task -- curriculum off, 3-8 obstacles, goals out to 3.5 m --
no matter what the policy was trained on.
"""

import argparse
import os
import time

import envs  # noqa: F401  (imported for its side effect: registers GoalNav-v0)
import gymnasium as gym
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from train_goal_nav import ENV_ID
from train_goal_nav import ENV_KWARGS
from train_goal_nav import MODEL_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("x", nargs="?", type=float, help="metres to the rover's right")
    p.add_argument("y", nargs="?", type=float, help="metres forward")
    p.add_argument("--episodes", type=int, help="run this many headless episodes instead")
    p.add_argument("--model", default=os.path.join(MODEL_DIR, "model"), help="policy to load")
    return p.parse_args()


def make_env():
    """Same observation space the policy trained with, but the full task: curriculum off."""
    kwargs = dict(ENV_KWARGS)
    kwargs["curriculum"] = False
    return gym.make(ENV_ID, **kwargs)


def goal_option(args):
    """`x y` on the command line is in the prompt's convention -- x metres right, y metres
    forward -- while the env's frame is (forward, left), hence the swap and the sign."""
    if args.x is None or args.y is None:
        return None
    return {"goal": (args.y, -args.x)}


def run_headless(model, env, episodes, options):
    outcomes = {"arrived": 0, "collided": 0, "timeout": 0}
    drifts, distances, lengths = [], [], []

    for _ in range(episodes):
        obs, _ = env.reset(options=options)
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        if info["arrived"]:
            outcomes["arrived"] += 1
        elif info["collided"]:
            outcomes["collided"] += 1
        else:
            outcomes["timeout"] += 1
        drifts.append(info["odometry_error"])
        distances.append(info["distance"])
        lengths.append(steps)

    print(f"\n{episodes} episodes on the full task:")
    for name, count in outcomes.items():
        print(f"  {name:9s} {count:5d}  ({count / episodes:6.1%})")
    print(f"  mean final distance to goal : {np.mean(distances):.2f} m")
    print(f"  mean odometry drift at end  : {np.mean(drifts):.2f} m")
    print(f"  mean episode length         : {np.mean(lengths):.0f} steps")


def run_viewer(model, env, options):
    # GoalNavEnv has no gym render_mode of its own (same pattern as evaluate_real.py), so the
    # viewer is driven directly off the underlying MuJoCo model/data.
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    obs, info = env.reset(options=options)
    print(f"Goal (forward, left) = {np.round(info['goal_command'], 2)} m")
    episode_reward = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            viewer.sync()

            if terminated or truncated:
                outcome = (
                    "ARRIVED"
                    if info["arrived"]
                    else "COLLIDED"
                    if info["collided"]
                    else "ran out of time"
                )
                print(
                    f"{outcome}: reward={episode_reward:.1f} "
                    f"final distance={info['distance']:.2f} m "
                    f"odometry drift={info['odometry_error']:.2f} m"
                )
                episode_reward = 0.0
                obs, info = env.reset(options=options)
                print(f"Goal (forward, left) = {np.round(info['goal_command'], 2)} m")

            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    args = parse_args()
    env = make_env()
    model = PPO.load(args.model, env=env)
    options = goal_option(args)

    if args.episodes:
        run_headless(model, env, args.episodes, options)
    else:
        run_viewer(model, env, options)
    env.close()


if __name__ == "__main__":
    main()
