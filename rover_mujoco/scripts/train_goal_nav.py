"""Trains a PPO policy on GoalNav-v0: drive to a commanded (x, y) offset inside a walled 5x5 m
arena without hitting the obstacles scattered around it. See docs/rl-goal-nav.md.

Same sim-to-real footing as train_real.py's line follower -- camera + noisy encoders + the
rover's three-beam lidar, wheel-velocity actions, and the same domain randomization ranges --
with the goal handed to the policy as a dead-reckoned vector that drifts, rather than as
privileged ground truth.

Dict observations (image + encoders + lidar + goal) need "MultiInputPolicy", not "CnnPolicy".
"""

import os

import envs  # noqa: F401  (imported for its side effect: registers GoalNav-v0)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "GoalNav-v0"
# Starting point, not a tuned number. The line follower needed 10M steps to handle full DR
# (see train_real.py), and this task is harder -- it has to avoid obstacles *and* correct for
# odometry drift -- so treat this as the first checkpoint to inspect, not the finish line.
TOTAL_TIMESTEPS = 10_000_000
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../runs/ppo_goal_nav")

# Rollouts are rendering/physics-bound, same as train_real.py -- see its comment for why 8.
N_ENVS = 8

# ent_coef: the line-follower sweep found the default 0.0 plateaued from under-exploration
# while 0.01 was still climbing at 3M steps (see sweep_hparams.py / continue_sweep_winner.py).
# This task needs more exploration, not less -- a policy that never turns never finds a goal
# behind it -- so start from that sweep's winner rather than from the SB3 default.
ENT_COEF = 0.01


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    model = PPO("MultiInputPolicy", env, ent_coef=ENT_COEF, verbose=1, tensorboard_log=MODEL_DIR)
    print(f"Training on device: {model.device}")

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(os.path.join(MODEL_DIR, "model"))
    print(f"Saved trained model to {MODEL_DIR}/model.zip")
    print(f"View training curves with: uv run tensorboard --logdir {MODEL_DIR}")


if __name__ == "__main__":
    main()
