import argparse
import os

import envs  # noqa: F401  (imported for its side effect: registers LineFollowerReal-v0)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb_logging import start_run

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="log to Weights & Biases as well as TensorBoard (see scripts/wandb_logging.py)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS, help="override the training budget"
    )
    parser.add_argument("--n-envs", type=int, default=N_ENVS, help="override worker count")
    parser.add_argument(
        "--run-name", default=None, help="subdirectory under runs/ so runs do not overwrite"
    )
    args = parser.parse_args()

    model_dir = (
        MODEL_DIR
        if args.run_name is None
        else os.path.join(os.path.dirname(MODEL_DIR), args.run_name)
    )

    os.makedirs(model_dir, exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=model_dir)
    print(f"Training on device: {model.device}")

    run, wandb_callback = (
        start_run(
            project="rover-line-follower",
            name=os.path.basename(model_dir),
            config={
                "env_id": ENV_ID,
                "total_timesteps": args.timesteps,
                "n_envs": args.n_envs,
                "env_kwargs": {},
            },
            model_dir=model_dir,
        )
        if args.wandb
        else (None, None)
    )
    callbacks = [wandb_callback] if wandb_callback is not None else None

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    model.save(os.path.join(model_dir, "model"))
    if run is not None:
        run.finish()
    print(f"Saved trained model to {model_dir}/model.zip")
    print(f"View training curves with: uv run tensorboard --logdir {model_dir}")


if __name__ == "__main__":
    main()
