"""Offline preparation and sensor conversions for the eleven-input PPO policy."""

import argparse
import hashlib
import json
from pathlib import Path

from envs.contract import MODEL_API
import numpy as np

REQUIRED_TRACKS = ("circle", "figure8", "oval")


def encoder_speeds(delta_counts, elapsed_s, counts_per_revolution, signs):
    """Convert signed encoder increments using the measured interval and output-shaft resolution."""
    counts = np.asarray(delta_counts, dtype=float)
    signs = np.asarray(signs, dtype=float)
    if (
        counts.shape != (2,)
        or signs.shape != (2,)
        or not np.isfinite(counts).all()
        or not np.isin(signs, [-1, 1]).all()
        or not np.isfinite(elapsed_s)
        or elapsed_s <= 0
        or not np.isfinite(counts_per_revolution)
        or counts_per_revolution <= 0
    ):
        raise ValueError(
            "Supply finite counts, a positive measured interval/resolution, and two encoder signs"
        )
    # Missed polls increase elapsed time instead of making the reported wheel speed spike.
    return counts * signs * (2 * np.pi / counts_per_revolution / elapsed_s)


def evaluation_evidence(directory, model_path, mode):
    """Verify per-episode outcomes and checkpoint identity before creating deployment evidence."""
    metadata = json.loads((directory / "evaluation_metadata.json").read_text())
    outcomes = json.loads((directory / "evaluation.json").read_text())
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if metadata["model_sha256"] != digest or metadata["dynamics"] != mode:
        raise ValueError(f"{directory} evaluates a different checkpoint or dynamics mode")
    rates, counts = {}, {}
    for track in REQUIRED_TRACKS:
        episodes = outcomes.get(track, {}).get("episodes", [])
        if len(episodes) < 20 or len({episode["seed"] for episode in episodes}) != len(episodes):
            raise ValueError(f"{mode}/{track} needs at least 20 distinct evaluation seeds")
        if any(episode.get("dynamics") != mode for episode in episodes):
            raise ValueError(f"{mode}/{track} has inconsistent dynamics provenance")
        # Count geometric completions so a shaped reward or a pooled rate cannot qualify a track.
        rates[track] = sum(
            episode["is_success"] is True
            and episode["reason"] == "completed"
            and 0 <= episode["max_deviation_cm"] <= 6
            for episode in episodes
        ) / len(episodes)
        counts[track] = len(episodes)
        if rates[track] < 0.9:
            raise ValueError(f"{mode}/{track} completed {rates[track]:.0%}, below 90%")
    return {"model_sha256": digest, "rates": rates, "episodes": counts}


def check_manifest(model_path, manifest_path):
    """Refuse physical execution unless both RL variants carry the required simulation evidence."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if manifest.get("model_api") != MODEL_API or manifest.get("model_sha256") != digest:
        raise ValueError("Deployment manifest does not match this policy and API")
    for mode in ("nominal", "bam", "dr"):
        evidence = manifest.get("evaluations", {}).get(mode, {})
        if mode != "nominal" and evidence.get("model_sha256") != digest:
            raise ValueError(f"{mode} evidence does not match this checkpoint")
        for track in REQUIRED_TRACKS:
            rate = evidence.get("rates", {}).get(track, -1)
            if not 0.9 <= rate <= 1 or evidence.get("episodes", {}).get(track, 0) < 20:
                raise ValueError(f"Missing qualified {mode}/{track} evaluation")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--nominal-model", type=Path, required=True)
    for mode in ("nominal", "bam", "dr"):
        parser.add_argument(f"--{mode}-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Preparation reads saved simulations and writes a local manifest without contacting the rover.
    evidence = {
        mode: evaluation_evidence(
            getattr(args, f"{mode}_evaluation"),
            args.nominal_model if mode == "nominal" else args.model,
            mode,
        )
        for mode in ("nominal", "bam", "dr")
    }
    manifest = {
        "model_api": MODEL_API,
        "model_sha256": evidence["dr"]["model_sha256"],
        "evaluations": evidence,
    }
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Simulation gate passed; deployment manifest: {args.output}")


if __name__ == "__main__":
    main()
