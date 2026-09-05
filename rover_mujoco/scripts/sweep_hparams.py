"""Small hyperparameter sweep for LineFollowerReal-v0, run after the 10M-step baseline
(runs/archive/model_10M_v6.zip) plateaued around ep_rew_mean~90-105 with no laps/finishes —
consistent, safe, but under-exploring. Each variant here is short (3M steps, ~35-45min at
our measured throughput) to compare learning curves cheaply before committing to a long run.

Usage: uv run python scripts/sweep_hparams.py <variant_name>
"""

import os
import sys

import envs  # noqa: F401  (imported for its side effect: registers LineFollowerReal-v0)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from train_real import ENV_ID
from train_real import N_ENVS

SWEEP_TIMESTEPS = 3_000_000
SWEEP_DIR = os.path.join(os.path.dirname(__file__), "../runs/sweep")

VARIANTS = {
    # Baseline PPO defaults reproduced here for reference (not meant to be re-run — the
    # 10M-step run already covers this point).
    "baseline": dict(),
    # SB3 default ent_coef=0.0 — no entropy bonus at all. This is the most likely fix for
    # "found one safe behavior and stopped exploring": reward the policy for keeping its
    # action distribution's entropy up, directly countering premature convergence.
    "entropy_0.01": dict(ent_coef=0.01),
    "entropy_0.02": dict(ent_coef=0.02),
    # Constant 3e-4 for the entire run may be too high once the policy's already found a
    # decent local behavior (causing it to keep perturbing away from good updates — matches
    # the elevated approx_kl/clip_fraction seen in the plateaued region). Linear decay to 0.
    "lr_linear_decay": dict(learning_rate=lambda progress_remaining: 3e-4 * progress_remaining),
    # Combine both.
    "entropy_0.01_lr_decay": dict(
        ent_coef=0.01,
        learning_rate=lambda progress_remaining: 3e-4 * progress_remaining,
    ),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print("Usage: uv run python scripts/sweep_hparams.py <variant>")
        print(f"Variants: {list(VARIANTS.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    kwargs = VARIANTS[name]
    out_dir = os.path.join(SWEEP_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=out_dir, **kwargs)
    print(f"Variant '{name}': {kwargs}, device={model.device}")

    model.learn(total_timesteps=SWEEP_TIMESTEPS)
    model.save(os.path.join(out_dir, "model"))
    print(f"Saved to {out_dir}/model.zip")


if __name__ == "__main__":
    main()
