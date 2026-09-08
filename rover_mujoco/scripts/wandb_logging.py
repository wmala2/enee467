"""Optional Weights & Biases logging for the training scripts.

SB3 ships a first-party W&B integration, so this is a thin wrapper: start a run, mirror the
TensorBoard scalars into it, and hand back a callback. It stays optional and off by default --
W&B is an external service, and a run uploads its metrics and config to wandb.ai.

    uv run python scripts/train_real.py --wandb                 # stream to wandb.ai
    WANDB_MODE=offline uv run python scripts/train_real.py --wandb   # log locally, sync later

Offline mode writes to ./wandb/ and never contacts the network; `wandb sync <dir>` uploads it
afterwards if you decide you want it. Use that if you would rather not put a training run on a
third-party service, or are working without a login.

Everything already logged to TensorBoard comes through, including the task-level metrics the
callbacks record -- arrival and collision rates, curriculum level, line-loss rate -- which are
the ones worth putting in a report. What W&B adds over TensorBoard is comparison across runs:
overlaying several policies on one axis, and sweeps with parallel-coordinate plots showing
which hyperparameters actually mattered.
"""

import os


def start_run(project, name, config, model_dir):
    """Begin a W&B run mirroring TensorBoard, returning (run, callback) or (None, None).

    Returns (None, None) rather than raising when wandb is missing or no credentials are
    available, so `--wandb` degrades to a normal run instead of losing the training."""
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except ImportError:
        print("wandb is not installed; add it with `uv add wandb`. Continuing without it.")
        return None, None

    if os.environ.get("WANDB_MODE") != "offline" and not (
        os.environ.get("WANDB_API_KEY") or os.path.exists(os.path.expanduser("~/.netrc"))
    ):
        print(
            "wandb has no credentials. Run `wandb login`, or set WANDB_MODE=offline to log\n"
            "locally and sync later. Continuing without it."
        )
        return None, None

    run = wandb.init(
        project=project,
        name=name,
        config=config,
        sync_tensorboard=True,  # pull in everything SB3 already writes
        monitor_gym=False,
        save_code=True,
    )
    callback = WandbCallback(
        gradient_save_freq=0,  # weights histograms are large and rarely worth the upload
        model_save_path=os.path.join(model_dir, "wandb_models"),
        verbose=1,
    )
    return run, callback
