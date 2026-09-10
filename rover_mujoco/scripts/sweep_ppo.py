"""Compare four PPO fine-tuning settings on frozen nominal scenes and fresh final test seeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TRACKS = ["circle", "figure8", "oval"]
VARIANTS = {
    "reference": {},
    "learning_rate": {"learning-rate": 0.001},
    "entropy": {"ent-coef": 0.0},
    "discount": {"gamma": 0.98},
}


def validation_score(entry):
    """Rank the weakest track first, using mean validation return only to break ties."""
    results = list(entry["tracks"].values())
    return min(r["success_rate"] for r in results), sum(r["mean_return"] for r in results) / len(
        results
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-episodes", type=int, default=50)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("runs") / time.strftime("ppo-sweep-%Y%m%d-%H%M%S")
    )
    args = parser.parse_args()
    if min(args.timesteps, args.final_episodes) < 1 or not args.checkpoint.is_file():
        parser.error("provide a saved checkpoint and positive step and episode counts")
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

    # Freeze code and assets so workspace edits cannot change conditions between candidates.
    package = Path(__file__).resolve().parents[1]
    snapshot = args.output / "snapshot"
    for directory in ("envs", "assets"):
        shutil.copytree(
            package / directory, snapshot / directory, ignore=shutil.ignore_patterns("__pycache__")
        )
    (snapshot / "scripts").mkdir()
    for name in ("train_ppo.py", "evaluate_ppo.py", "plot_ppo_training.py", "sweep_ppo.py"):
        shutil.copy2(package / "scripts" / name, snapshot / "scripts" / name)
    # Freeze the starting weights and lineage alongside the environment snapshot.
    source = snapshot / "source"
    source.mkdir()
    checkpoint = source / args.checkpoint.name
    shutil.copy2(args.checkpoint, checkpoint)
    source_directory = args.checkpoint.parent
    if source_directory.name == "checkpoints":
        source_directory = source_directory.parent
    if (source_directory / "config.json").exists():
        shutil.copy2(source_directory / "config.json", source / "config.json")
    manifest = {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    config = {
        **vars(args),
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "tracks": TRACKS,
        "reward_centering": "camera",
        "n_envs": 6,
        "variants": {name: VARIANTS[name] for name in args.variants},
        "validation_seed": 1000,
        "benchmark_seed": 10000,
        "final_seed": 20000,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONPATH": str(snapshot),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    summary = {}
    for name in args.variants:
        # Every candidate starts from the same checkpoint and changes at most one hyperparameter.
        output = args.output / name
        command = [
            sys.executable,
            str(snapshot / "scripts/train_ppo.py"),
            "--resume",
            str(checkpoint),
            "--output",
            str(output),
            "--timesteps",
            str(args.timesteps),
            "--seed",
            str(args.seed),
            "--tracks",
            *TRACKS,
            "--reward-centering",
            "camera",
            "--wandb-group",
            args.output.name,
        ]
        for parameter, value in VARIANTS[name].items():
            command.extend([f"--{parameter}", str(value)])
        if args.no_wandb:
            command.append("--no-wandb")
        print(f"Starting {name}: {VARIANTS[name]}", flush=True)
        with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        subprocess.run(
            [sys.executable, str(snapshot / "scripts/plot_ppo_training.py"), str(output)],
            env=environment,
            check=True,
        )
        history = json.loads((output / "validation.json").read_text(encoding="utf-8"))
        best = max(history, key=validation_score)
        summary[name] = {
            "validation_score": validation_score(best),
            "selected_timesteps": best["timesteps"],
            "validation": best["tracks"],
            "benchmark": json.loads(
                (output / "evaluation/evaluation.json").read_text(encoding="utf-8")
            ),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Finished {name}: validation score {validation_score(best)}", flush=True)

    # Select using validation, then test the winner on untouched seeds.
    winner = max(summary, key=lambda name: summary[name]["validation_score"])
    model_path = args.output / winner / "best_model.zip"
    final_output = args.output / "final_evaluation"
    subprocess.run(
        [
            sys.executable,
            str(snapshot / "scripts/evaluate_ppo.py"),
            str(model_path),
            "--output",
            str(final_output),
            "--seed",
            "20000",
            "--episodes",
            str(args.final_episodes),
            "--tracks",
            *TRACKS,
            "--video",
        ],
        env=environment,
        check=True,
    )
    results = json.loads((final_output / "evaluation.json").read_text(encoding="utf-8"))
    decision = {
        "winner": winner,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "selection": "minimum track validation success, then mean validation return",
        "final_seed": 20000,
        "episodes_per_track": args.final_episodes,
        "rates": {track: result["success_rate"] for track, result in results.items()},
        "required_success_rate": 0.9,
        "passed": all(result["success_rate"] >= 0.9 for result in results.values()),
    }
    (args.output / "winner.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if not args.no_wandb:
        import wandb

        # Log the final test as a separate grouped run without feeding it back into selection.
        with wandb.init(
            project="rover-line-follower",
            group=args.output.name,
            name=f"{args.output.name}-evaluation",
            job_type="evaluation",
            config=decision,
            dir=str(args.output),
        ) as run:
            for track, result in results.items():
                run.summary[f"test/{track}/success_rate"] = result["success_rate"]
                run.summary[f"test/{track}/mean_deviation_cm"] = result["mean_deviation_cm"]
            run.log({"test/trajectories": wandb.Image(str(final_output / "trajectories.png"))})
            artifact = wandb.Artifact(args.output.name, type="model")
            for path in (
                model_path,
                args.output / "winner.json",
                args.output / "summary.json",
                final_output / "evaluation.json",
            ):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
    print(
        f"Winner: {winner}; fresh test rates: {decision['rates']}; passed={decision['passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
