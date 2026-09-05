import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import envs  # Registers LineFollowerReal-v0

# LineFollowerReal-v0: the sim-to-real-constrained line follower — real observation space
# (camera + noisy wheel encoders, no ground truth), real action space (left/right wheel
# velocity), and domain randomization over motor lag, action noise/latency, encoder noise,
# and wheel friction on top of LineFollower-v0's visual DR. See docs/rl-line-follower.md.
#
# Dict observations (image + encoders) need "MultiInputPolicy", not "CnnPolicy".
ENV_ID = "LineFollowerReal-v0"
TOTAL_TIMESTEPS = 10_000_000  # 2M centered on the line reliably but didn't drive forward
# consistently or handle domain randomization well (see training-run history) — vision-based
# PPO with heavy DR typically needs this much more, and it's cheap at our measured throughput
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../runs/ppo_line_follower_real")

# Rollout collection (MuJoCo rendering + physics) is CPU-bound and dominates wall-clock
# time, not the policy's forward/backward pass — so this is parallelized across N_ENVS
# subprocesses (SB3 measured on this machine: throughput roughly doubles from 1->8 envs,
# then flattens out, likely rendering-bound rather than CPU-bound, so going higher wasn't
# worth the complexity). "fork" start method so each subprocess inherits the "envs"
# registration already imported above; PPO itself picks the GPU automatically
# (device="auto") when one's available (checked via nvidia-smi) for the actual gradient
# updates.
N_ENVS = 8

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = make_vec_env(
        ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=MODEL_DIR)
    print(f"Training on device: {model.device}")

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(os.path.join(MODEL_DIR, "model"))
    print(f"Saved trained model to {MODEL_DIR}/model.zip")
    print(f"View training curves with: uv run tensorboard --logdir {MODEL_DIR}")

if __name__ == "__main__":
    main()
