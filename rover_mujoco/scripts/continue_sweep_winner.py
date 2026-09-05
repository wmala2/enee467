"""Continue training the entropy_0.01 sweep winner (runs/sweep/entropy_0.01/model.zip) further
— its reward curve was still climbing at 3M steps (86.7 -> 191, no plateau), unlike the
plateaued 10M-step ent_coef=0.0 baseline. Resumes from that checkpoint rather than restarting,
so the already-good 3M steps aren't wasted.
"""

import os

import envs  # noqa: F401  (imported for its side effect: registers LineFollowerReal-v0)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from train_real import ENV_ID
from train_real import N_ENVS

ADDITIONAL_TIMESTEPS = 7_000_000  # 3M already done -> 10M total, matching the baseline's scale
OUT_DIR = os.path.join(os.path.dirname(__file__), "../runs/sweep/entropy_0.01_continued")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    model = PPO.load(
        os.path.join(os.path.dirname(__file__), "../runs/sweep/entropy_0.01/model.zip"),
        env=env,
        tensorboard_log=OUT_DIR,
    )
    print(
        f"Resumed at {model.num_timesteps} timesteps, ent_coef={model.ent_coef}, device={model.device}"
    )
    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=False)
    model.save(os.path.join(OUT_DIR, "model"))
    print(f"Saved to {OUT_DIR}/model.zip, total timesteps={model.num_timesteps}")


if __name__ == "__main__":
    main()
