import os

import envs  # noqa: F401  (imported for its side effect: registers LineFollower-v0)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# LineFollower-v0: camera-only line follower with domain randomization, the actual
# sim-to-real-facing RL task this repo trains (see docs/rl-line-follower.md). PPO is used
# because it trains well on CPU-only hardware. Camera-image observations make each
# environment step much slower than a vector-obs task — rendering dominates — so expect
# roughly an order of magnitude fewer steps/sec than a vector-observation env would give.
ENV_ID = "LineFollower-v0"
TOTAL_TIMESTEPS = 50_000
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../runs/ppo_line_follower")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # domain_randomize=True (the env default): floor/line color, light intensity, and
    # camera mount pose all get jittered every reset, so the policy sees many plausible
    # renderings of "a rover with a downward camera" instead of overfitting to one exact
    # scene — see docs/rl-line-follower.md for why this matters for sim-to-real transfer.
    env = Monitor(gym.make(ENV_ID))
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=MODEL_DIR)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(os.path.join(MODEL_DIR, "model"))
    print(f"Saved trained model to {MODEL_DIR}/model.zip")
    print(f"View training curves with: uv run tensorboard --logdir {MODEL_DIR}")


if __name__ == "__main__":
    main()
