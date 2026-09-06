"""Watches a trained GoalNav-v0 policy drive to its goal in the MuJoCo viewer, and reports
whether it arrived, hit something, or ran out of time -- plus how far its dead-reckoned
estimate had drifted from the truth by then.

    uv run python scripts/evaluate_goal_nav.py            # random goals
    uv run python scripts/evaluate_goal_nav.py 1 2        # "drive to (1, 2)": 1 m right,
                                                          # 2 m forward
"""

import os
import sys
import time

import envs  # noqa: F401  (imported for its side effect: registers GoalNav-v0)
import gymnasium as gym
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from train_goal_nav import ENV_ID
from train_goal_nav import MODEL_DIR


def parse_goal(argv):
    """`x y` on the command line, in the prompt's own convention: x is metres to the rover's
    right, y is metres forward. The env's frame is (forward, left), hence the swap and sign."""
    if len(argv) < 3:
        return None
    right, forward = float(argv[1]), float(argv[2])
    return (forward, -right)


def main():
    goal = parse_goal(sys.argv)
    # GoalNavEnv has no gym render_mode of its own (same pattern as evaluate_real.py), so the
    # viewer is driven directly off the underlying MuJoCo model/data.
    env = gym.make(ENV_ID)
    model = PPO.load(os.path.join(MODEL_DIR, "model"), env=env)

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    options = {"goal": goal} if goal else None
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

    env.close()


if __name__ == "__main__":
    main()
