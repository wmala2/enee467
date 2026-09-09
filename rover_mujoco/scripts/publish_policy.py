"""Publish a trained policy to the Hugging Face Hub, but only if it passes the gate.

    uv run python scripts/publish_policy.py runs/my-run --repo user/rover-line-follower-ppo
    uv run python scripts/publish_policy.py runs/my-run --repo ... --dry-run

rover_control/rl_rover.py downloads whatever sits in that repo and feeds it straight to the
motors, so a policy whose observation layout does not match the runtime's does not fail at a
boundary, it fails on the robot. Everything here exists to make that impossible to do by
accident: the shapes are checked against envs/contract.py, the policy is actually stepped in
the environment, and it must carry evidence that it met its success criterion. Refusing to
publish is the cheap outcome.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import envs  # noqa: F401  (registers the environments)
from envs import contract
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def git_provenance(repo_root):
    """Commit, branch, and whether the tree was dirty when this was published."""

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def check_contract(model):
    """The policy's spaces must be exactly what the runtime will hand it."""
    observation = model.observation_space
    action = model.action_space
    problems = []
    if observation.shape != (contract.OBS_LEN,):
        problems.append(f"observation is {observation.shape}, contract says ({contract.OBS_LEN},)")
    if action.shape != (contract.ACTION_LEN,):
        problems.append(f"action is {action.shape}, contract says ({contract.ACTION_LEN},)")
    if problems:
        raise SystemExit(
            "contract check failed:\n  "
            + "\n  ".join(problems)
            + "\n\nrover_control/rl_rover.py builds observations to this contract; publishing "
            "a policy that disagrees breaks the rover rather than this script."
        )


def smoke_run(model, steps=50):
    """Step the policy in the real environment, so a loadable-but-broken policy is caught."""
    environment = gym.make("LineFollowerPPO-v0", track="circle")
    try:
        observation, _ = environment.reset(seed=0)
        for step in range(steps):
            action, _ = model.predict(observation, deterministic=True)
            if not np.all(np.isfinite(action)):
                raise SystemExit(f"policy produced a non-finite action at step {step}: {action}")
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                observation, _ = environment.reset(seed=step)
    finally:
        environment.close()


def load_qualification(run_dir, required):
    """A policy publishes only with evidence that it met the success criterion."""
    candidates = sorted(run_dir.glob("evaluation*/qualification.json"))
    if not candidates:
        raise SystemExit(
            f"no evaluation*/qualification.json under {run_dir}. Run scripts/evaluate_ppo.py "
            "and record the result before publishing."
        )
    # Prefer the record covering the most episodes per track.
    best = max(candidates, key=lambda p: json.loads(p.read_text()).get("episodes_per_track", 0))
    qualification = json.loads(best.read_text())
    failed = {t: r for t, r in qualification["rates"].items() if r < required}
    if failed:
        raise SystemExit(f"{best} reports rates below {required:.0%}: {failed}")
    return best, qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="training run directory")
    parser.add_argument("--repo", help="Hugging Face repo id, e.g. user/rover-line-follower-ppo")
    parser.add_argument("--model", default="best_model.zip")
    parser.add_argument("--required-rate", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", help="run the gate, upload nothing")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_path = args.run / args.model
    if not model_path.is_file():
        raise SystemExit(f"no model at {model_path}")

    model = PPO.load(model_path, device="cpu")
    check_contract(model)
    smoke_run(model)
    qualification_path, qualification = load_qualification(args.run, args.required_rate)

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if qualification.get("model_sha256") not in (None, digest):
        raise SystemExit(
            f"{qualification_path} qualifies a different checkpoint "
            f"({qualification['model_sha256'][:12]}) than {model_path.name} ({digest[:12]})"
        )

    manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "model_api": contract.MODEL_API,
        "model_file": args.model,
        "model_sha256": digest,
        "obs_len": contract.OBS_LEN,
        "obs_layout": contract.OBS_LAYOUT,
        "action_len": contract.ACTION_LEN,
        "action_layout": contract.ACTION_LAYOUT,
        "control_hz": contract.CONTROL_HZ,
        "qualification": qualification,
        "git": git_provenance(repo_root),
        "config": json.loads((args.run / "config.json").read_text())
        if (args.run / "config.json").is_file()
        else None,
    }
    manifest_path = args.run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"contract OK      : obs {contract.OBS_LEN}, action {contract.ACTION_LEN}")
    print("smoke run OK     : 50 steps, finite actions")
    print(f"qualified        : {qualification_path.relative_to(args.run)} {qualification['rates']}")
    print(f"model sha256     : {digest}")
    print(f"git              : {manifest['git']['commit']} dirty={manifest['git']['dirty']}")
    print(f"manifest         : {manifest_path}")

    if manifest["git"]["dirty"]:
        print(
            "\nWARNING: the working tree is dirty, so this checkpoint is not reproducible "
            "from the recorded commit."
        )
    if args.dry_run or not args.repo:
        print("\ndry run: nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo=args.model, repo_id=args.repo)
    api.upload_file(
        path_or_fileobj=str(manifest_path), path_in_repo="manifest.json", repo_id=args.repo
    )
    print(f"\npublished to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
