"""Export the TensorBoard scalars mirrored to W&B as a shareable training figure."""

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def save_training_curves(run_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Merge scalar events from the run's TensorBoard writers without smoothing away failures.
    scalars = {}
    for directory in sorted((run_directory / "tensorboard").glob("PPO_*")):
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            scalars.setdefault(tag, []).extend(
                (event.step, event.value) for event in events.Scalars(tag)
            )
    panels = [
        ("rollout/ep_rew_mean", "Training episode reward"),
        ("task/is_success", "Training completion rate"),
        ("eval/", "Validation completion by track"),
        ("task/mean_deviation_cm", "Training mean deviation (cm)"),
        ("train/entropy_loss", "Policy entropy loss"),
        ("train/approx_kl", "Approximate KL divergence"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (tag, title) in zip(axes.flat, panels):
        keys = (
            [tag]
            if tag != "eval/"
            else [
                key for key in scalars if key.startswith("eval/") and key.endswith("success_rate")
            ]
        )
        for key in keys:
            if key in scalars:
                points = sorted(scalars[key])
                axis.plot([p[0] for p in points], [p[1] for p in points], label=key)
        axis.set(title=title, xlabel="environment steps")
        axis.grid(alpha=0.2)
        if tag == "eval/":
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(run_directory / "training_curves.png", dpi=160)
    plt.close(figure)
    # Keep numerical values alongside the plot for later comparisons and reports.
    with (run_directory / "training_scalars.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "step", "value"])
        for tag, points in sorted(scalars.items()):
            writer.writerows((tag, step, value) for step, value in sorted(points))
    return run_directory / "training_curves.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    print(save_training_curves(parser.parse_args().run_directory))
