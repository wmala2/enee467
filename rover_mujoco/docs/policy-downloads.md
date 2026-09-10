# Download and run saved policies

The saved policies are available in the private
[Hugging Face backup](https://huggingface.co/CursedRock17/rover-line-follower-ppo/tree/main/backups/2026-09-10).
For the existing training and evaluation scripts, download the checkpoint and its
adjacent `config.json`, then pass the local ZIP path as usual.
The configuration preserves the dynamics mode, DR ranges, reward mode, and tracks;
the checkpoint alone does not tell the evaluation script which simulation to build.

Run these commands from `rover_mujoco` after `uv run hf auth login` if this machine
is not already authenticated with an account that can access the private repository.

| Policy | `HF_POLICY_RUN` |
| --- | --- |
| Qualified DR/BAM policy | `dr-bam-bringup/ppo-dr-seed0` |
| Qualified nominal reference | `ppo-continued-seed0` |
| Nominal sweep's selected policy | `ppo-hparam-sweep-seed0/reference` |

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

Change `HF_POLICY_RUN` to another row before downloading to use its checkpoint;
change `--tracks` or `--seed` to explore other tracks or episode conditions.
For headless evaluation, use `MUJOCO_GL=egl` and omit `--viewer`.
The same local ZIP path works with `train_ppo.py --resume`, with the desired
training dynamics selected explicitly through that script's options.
The CLI preserves the repository's nested directory structure under `runs/hf`.

## Loading from Python

Python can download on demand using `hf_hub_download` or `snapshot_download`,
then give the returned local path to `PPO.load`.
Hugging Face handles authentication, versioned caching, and downloads; SB3 still
loads its ZIP checkpoint rather than accepting a Hub repository ID directly.
[Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
and [SB3 checkpoint format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)
describe those two parts of the process.

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

After the files are cached, `snapshot_download(..., local_files_only=True)` can
resolve them without network access.
Use an explicit output directory outside the shared HF cache if passing a cached
path to `evaluate_ppo.py`, because that script otherwise writes results beside
the checkpoint.
The archive also contains historical policies with older interfaces; the table
above identifies checkpoints compatible with the current PPO evaluation script.
Physical deployment additionally requires the matching `deployment.json` and
measured encoder calibration described in the [DR/BAM guide](dr-bam-line-follower.md#preparing-physical-rollout).
