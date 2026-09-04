import os
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

import envs  # Registers LineFollowerReal-v0
from train_real import ENV_ID, MODEL_DIR

def main():
    # LineFollowerRealEnv has no gym render_mode support of its own (see teleop_rover.py for
    # the same pattern), so the viewer is driven directly off the underlying MuJoCo model/data.
    env = gym.make(ENV_ID)
    model = PPO.load(os.path.join(MODEL_DIR, "model"), env=env)

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    obs, info = env.reset()
    episode_reward = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            viewer.sync()

            if terminated or truncated:
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0.0
                obs, info = env.reset()

            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    env.close()

if __name__ == "__main__":
    main()
