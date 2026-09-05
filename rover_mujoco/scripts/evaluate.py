import os
import time

import envs  # noqa: F401  (imported for its side effect: registers LineFollower-v0)
import gymnasium as gym
import mujoco.viewer
from stable_baselines3 import PPO
from train import ENV_ID
from train import MODEL_DIR


def main():
    # LineFollowerEnv has no gym render_mode support of its own (see teleop_rover.py for
    # the same pattern), so the viewer is driven directly off the underlying MuJoCo model/data.
    env = gym.make(ENV_ID)
    model = PPO.load(os.path.join(MODEL_DIR, "model"), env=env)

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    obs, _ = env.reset()
    episode_reward = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Trained policy instead of train.py's exploration noise: deterministic=True
            # picks the model's best action rather than sampling from its distribution.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            viewer.sync()

            if terminated or truncated:
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0.0
                obs, _ = env.reset()

            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    env.close()


if __name__ == "__main__":
    main()
