"""Rolls out the sim-to-real line-following policy in simulation -- in the viewer, or as a
batch of headless episodes that reports whether it actually completes tracks.

    uv run python scripts/evaluate_real.py                     # viewer, local runs/ model
    uv run python scripts/evaluate_real.py --hub               # viewer, the deployed weights
    uv run python scripts/evaluate_real.py --hub --episodes 30 # headless, success statistics

`--hub` loads the exact weights rover_control/rl_rover.py downloads and runs on the physical
rover. That matters: without it this script reads runs/ppo_line_follower_real/ while deployment
reads the Hub, so it is possible to ship a policy to hardware that was never rolled out in sim.
Evaluate what you are about to deploy.
"""

import argparse
import os
import time

import envs  # noqa: F401  (imported for its side effect: registers LineFollowerReal-v0)
import gymnasium as gym
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from train_real import ENV_ID
from train_real import MODEL_DIR

HF_REPO_ID = "CursedRock17/rover-line-follower-ppo"  # same repo rl_rover.py deploys from


def load_model(use_hub, env):
    if use_hub:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.zip")
        print(f"loaded deployed weights from {HF_REPO_ID}")
        return PPO.load(path, env=env)
    return PPO.load(os.path.join(MODEL_DIR, "model"), env=env)


def episode_outcome(env):
    """LineFollowerRealEnv returns an empty info dict, so read the episode's own state. A track
    is 'completed' when a closed track has been lapped or an open one driven end to end."""
    u = env.unwrapped
    return {
        "completed": bool(u._laps_completed > 0 or u._finished),
        "progress": float(u._unwrapped_progress / u._path_total_len),
        "lost_line": u._lost_steps >= 1,
        "tipped": bool(u.data.qpos[2] < 0.03),
        "steps": u._episode_steps,
    }


def run_headless(model, env, episodes):
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        out = episode_outcome(env)
        out["reward"] = total
        results.append(out)

    done = sum(r["completed"] for r in results)
    tipped = sum(r["tipped"] for r in results)
    print(f"\n{episodes} episodes on {ENV_ID}:")
    print(f"  completed a lap/traversal : {done:3d}  ({done / episodes:6.1%})")
    print(f"  tipped over               : {tipped:3d}  ({tipped / episodes:6.1%})")
    print(f"  mean track progress       : {np.mean([r['progress'] for r in results]):6.1%}")
    print(f"  mean episode reward       : {np.mean([r['reward'] for r in results]):7.1f}")
    print(f"  mean episode length       : {np.mean([r['steps'] for r in results]):5.0f} steps")


def run_viewer(model, env):
    # LineFollowerRealEnv has no gym render_mode of its own (see teleop_rover.py for the same
    # pattern), so the viewer is driven directly off the underlying MuJoCo model/data.
    mj_model, mj_data = env.unwrapped.model, env.unwrapped.data
    obs, _ = env.reset()
    episode_reward = 0.0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    # MuJoCo's viewer shows geom groups 0-2 by default, and the track geoms are group 3 -- so
    # without this the rover appears to drive on bare floor while following a line only it can
    # see. The policy's own offscreen renderer has always shown every group, which is why
    # headless evaluation scores fine while the viewer looks broken.
    for group in (3, 4):
        viewer.opt.geomgroup[group] = 1
        while viewer.is_running():
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            viewer.sync()
            if terminated or truncated:
                out = episode_outcome(env)
                print(
                    f"{'COMPLETED' if out['completed'] else 'ended'}: "
                    f"reward={episode_reward:.1f} progress={out['progress']:.1%} "
                    f"steps={out['steps']}"
                )
                episode_reward = 0.0
                obs, _ = env.reset()
            remaining = mj_model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub", action="store_true", help="load the weights rl_rover.py deploys")
    p.add_argument("--episodes", type=int, help="run this many headless episodes instead")
    args = p.parse_args()

    env = gym.make(ENV_ID)
    model = load_model(args.hub, env)
    if args.episodes:
        run_headless(model, env, args.episodes)
    else:
        run_viewer(model, env)
    env.close()


if __name__ == "__main__":
    main()
