import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import envs  # Registers LineFollowerReal-v0

# LineFollowerReal-v0: the sim-to-real-constrained line follower — real observation space
# (camera + noisy wheel encoders, no ground truth), real action space (left/right wheel
# velocity), and domain randomization over motor lag, action noise/latency, encoder noise,
# and wheel friction on top of LineFollower-v0's visual DR. See docs/rl-line-follower.md.
#
# Dict observations (image + encoders) need "MultiInputPolicy", not "CnnPolicy".
ENV_ID = "LineFollowerReal-v0"
TOTAL_TIMESTEPS = 50_000
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../runs/ppo_line_follower_real")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = Monitor(gym.make(ENV_ID))
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=MODEL_DIR)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(os.path.join(MODEL_DIR, "model"))
    print(f"Saved trained model to {MODEL_DIR}/model.zip")
    print(f"View training curves with: uv run tensorboard --logdir {MODEL_DIR}")

if __name__ == "__main__":
    main()
