# Download and run saved policies

## What this is

Training a good policy from scratch takes hours. Instead of always starting over, you can
download one of the already-trained, already-verified policies from this project's private
Hugging Face backup and drop it straight into the simulator (or the real rover). This doc covers
picking a checkpoint, downloading it, and loading it back up.

All commands below are run from `rover_mujoco`.

## Step 1: Log in to Hugging Face

The backup repository is private, so you need an account with access before you can download
anything:

```bash
uv run hf auth login
```

## Step 2: Pick a policy

Each row below is a different trained policy, checked in at the same
[Hugging Face backup](https://huggingface.co/CursedRock17/rover-line-follower-ppo/tree/main/backups/2026-09-10).

| Policy | `HF_POLICY_RUN` |
| --- | --- |
| Qualified DR/BAM policy | `dr-bam-bringup/ppo-dr-seed0` |
| Qualified nominal reference | `ppo-continued-seed0` |
| Nominal sweep's selected policy | `ppo-hparam-sweep-seed0/reference` |

## Step 3: Download it and watch it run

Every checkpoint (`best_model.zip`) has an adjacent `config.json` that records exactly how it was
trained — the dynamics mode, domain-randomization ranges, reward mode, and tracks. Download both
together and keep them side by side; the evaluation script needs the config to rebuild a matching
simulation.

```bash
# Select a saved run and pin its download to the verified backup commit.
HF_POLICY_RUN=dr-bam-bringup/ppo-dr-seed0
HF_POLICY_SUBDIR="backups/2026-09-10/runs/$HF_POLICY_RUN"
uv run hf download CursedRock17/rover-line-follower-ppo \
  "$HF_POLICY_SUBDIR/best_model.zip" "$HF_POLICY_SUBDIR/config.json" \
  --revision 77c0bf85aa4c0dd36ec0862fd17be485c1e686ac \
  --local-dir runs/hf

# View the downloaded policy with dynamics recovered from its adjacent configuration.
MUJOCO_GL=glfw uv run scripts/evaluate_ppo.py \
  "runs/hf/$HF_POLICY_SUBDIR/best_model.zip" \
  --tracks figure8 --episodes 1 --seed 40000 --viewer
```

A MuJoCo window should open and show the rover driving the figure-eight track using the
downloaded policy. From here:

- Swap `HF_POLICY_RUN` for another row in the table to try a different policy.
- Change `--tracks` or `--seed` to see it handle a different track or starting condition.
- Drop `--viewer` and set `MUJOCO_GL=egl` to evaluate headlessly (no window, faster, works
  without a display).
- The same downloaded ZIP also works as a starting point for more training, via
  `train_ppo.py --resume` (pick the matching training dynamics through that script's own flags).

## Loading a policy from Python

If you're writing your own script instead of using `evaluate_ppo.py`, you can download and load a
policy in a few lines: `snapshot_download` fetches (and caches) the files from Hugging Face, and
`PPO.load` from Stable-Baselines3 reads the resulting ZIP directly — it doesn't understand Hub
repository IDs on its own, so the download step always comes first.

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from stable_baselines3 import PPO

# Cache the matching checkpoint and configuration from the same immutable revision.
subdir = "backups/2026-09-10/runs/dr-bam-bringup/ppo-dr-seed0"
snapshot = Path(
    snapshot_download(
        repo_id="CursedRock17/rover-line-follower-ppo",
        revision="77c0bf85aa4c0dd36ec0862fd17be485c1e686ac",
        allow_patterns=[f"{subdir}/best_model.zip", f"{subdir}/config.json"],
    )
)

# Load the cached policy for inference using the project's eleven-value sensor contract.
model = PPO.load(snapshot / subdir / "best_model.zip", device="cpu")
```

See the [Hugging Face download docs](https://huggingface.co/docs/huggingface_hub/guides/download)
and the [SB3 checkpoint format docs](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)
for more on how those two pieces work.

## Good to know

- Once a checkpoint is cached, `snapshot_download(..., local_files_only=True)` reloads it without
  hitting the network again.
- If you pass a cached path straight to `evaluate_ppo.py`, point it at an output directory outside
  the shared Hugging Face cache — otherwise it writes its results right next to the checkpoint.
- The backup repository also holds older, incompatible policy formats from earlier in the
  project; only the checkpoints listed in the table above match the current evaluation script.
- Deploying a policy on the physical rover needs one more file beyond the checkpoint and config:
  a `deployment.json` with measured encoder calibration, described in the
  [DR/BAM guide](dr-bam-line-follower.md#preparing-physical-rollout).
