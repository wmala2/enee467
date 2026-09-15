# Sim Line Follower (RL)

This example trains a PPO policy to follow the same camera-only line-following task as the PID baseline, then evaluates the resulting checkpoint with independent lap-completion metrics — it's how you get a learned controller to compare against (or eventually replace) hand-tuned PID gains. Both scripts target the Gymnasium environment registered as `LineFollowerPPO-v0` in `rover_mujoco/envs/__init__.py`.

## Prerequisites

- A `uv sync --extra cpu` (or `--extra cu121` on an NVIDIA machine) from the repository root — this pulls in `stable-baselines3`, `torch`, `gymnasium`, and `matplotlib` alongside MuJoCo. Note that both scripts hardcode `device="cpu"` for the PPO model, so a GPU doesn't speed up this particular training loop (small MLP policies train faster on CPU with SB3).
- `MUJOCO_GL=egl` for headless training/evaluation, since `train_ppo.py` spawns several parallel MuJoCo worker processes (`--n-envs`, default 6) that each render a camera image without a display.
- Weights & Biases logging is **on by default** (project `rover-line-follower`) and training will fail to start if it can't authenticate — pass `--no-wandb` for local runs without a W&B account.
- A trained checkpoint (`best_model.zip`) to run `evaluate_ppo.py` on — either one you produce with `train_ppo.py`, or a saved policy from the Hugging Face backup described in `rover_mujoco/docs/policy-downloads.md`.

## Run it

```bash
# Train on the closed tracks and log to your authenticated W&B account.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run rover_mujoco/scripts/train_ppo.py --tracks circle figure8 oval

# Evaluate a checkpoint: watch one deterministic lap per track in the MuJoCo viewer.
uv run rover_mujoco/scripts/evaluate_ppo.py runs/ppo-20260101-000000/best_model.zip --episodes 1 --viewer
```

Training prints a periodic validation line (per-track completion rate) to the console and W&B, saves `best_model`/`final_model`/checkpoints under `--output` (default `runs/ppo-<timestamp>`), and finishes with a held-out test evaluation; `evaluate_ppo.py` prints one completion-rate/mean-error line per track and writes a JSON summary, CSV trajectories, and a trajectory plot to its output directory.

## How it works

### Training (`train_ppo.py`)

`make_environment` is the factory each parallel worker calls to build its own copy of the task. Each worker is pinned to one track and wrapped in SB3's `Monitor` so episode statistics get logged automatically; `random_start=True` means every episode begins at a different point along that track.

```python
def make_environment(track, reward_centering, dynamics="nominal", dr_ranges=None) -> gym.Env:
    # Each worker has a fixed track and samples start positions along it at reset.
    return Monitor(
        gym.make(
            ENV_ID,
            track=track,
            random_start=True,
            reward_centering=reward_centering,
            dynamics=dynamics,
            dr_ranges=dr_ranges,
        )
    )
```

A raw training-reward curve doesn't tell you whether the policy can actually finish a lap — reward is a shaped proxy, not the real success criterion. `CompletionCallback` periodically pauses training to run real evaluation episodes and picks the best checkpoint using each track's *completion rate* first, breaking ties with mean return, so a policy that overfits reward on one easy track can't hide a failure on another.

```python
    def _evaluate(self):
        results = evaluate(
            self.model,
            episodes=self.episodes,
            seed=1000,
            reward_centering=self.reward_centering,
            tracks=self.tracks,
            dynamics=self.dynamics,
            dr_ranges=self.dr_ranges,
        )
        for track, result in results.items():
            for key in ("success_rate", "mean_return", "mean_deviation_cm", "mean_progress"):
                self.logger.record(f"eval/{track}/{key}", result[key])
        score = (
            min(result["success_rate"] for result in results.values()),
            float(np.mean([result["mean_return"] for result in results.values()])),
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.output / "best_model")
```

`main` builds one SB3 `PPO` model (or resumes one with `--resume`) over `SubprocVecEnv`, which runs each worker's MuJoCo instance in its own OS process — physics simulation doesn't release Python's GIL, so true parallelism needs separate processes, not just threads.

```python
        factories: list[Callable[[], gym.Env]] = [
            partial(
                make_environment,
                args.tracks[i % len(args.tracks)],
                args.reward_centering,
                args.dynamics,
                dr_ranges,
            )
            for i in range(args.n_envs)
        ]
        environment = SubprocVecEnv(factories, start_method="spawn")
        environment.seed(args.seed)
```

Once created, the model just calls SB3's built-in training loop; the callbacks list is where this script's own logic (completion-based checkpointing, periodic saves) hooks into that loop.

```python
        callbacks: list[BaseCallback] = [
            CompletionCallback(
                args.output,
                args.eval_every,
                args.eval_episodes,
                args.reward_centering,
                args.tracks,
                args.dynamics,
                dr_ranges,
            ),
            CheckpointCallback(
                save_freq=max(1, args.eval_every // args.n_envs),
                save_path=str(args.output / "checkpoints"),
            ),
        ]
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
```

### Evaluation (`evaluate_ppo.py`)

`evaluate` is the function both `evaluate_ppo.py`'s CLI *and* `train_ppo.py`'s callback call — sharing one implementation means "how well is this policy doing" is measured identically during and after training. For each track it builds a fresh, single (non-parallel) environment and runs the requested number of episodes with fixed seeds, so results are reproducible.

```python
    for track in TRACKS if tracks is None else tracks:
        environment = gym.make(
            "LineFollowerPPO-v0",
            track=track,
            render_mode="human" if human else None,
            reward_centering=reward_centering,
            dynamics=dynamics,
            dr_ranges=dr_ranges,
        )
```

Unlike training, evaluation always picks the policy's single best-guess action (`deterministic=True`) rather than sampling from its probability distribution — you want to measure how the policy actually behaves when deployed, not its exploration noise.

```python
                while True:
                    start = time.monotonic()
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = environment.step(action)
                    total_reward += float(reward)
```

After every episode, the per-track outcomes are boiled down into a success rate and mean/max deviation — the same "did it actually finish the lap" metrics used for the PID baseline, so PPO and PID results are directly comparable.

```python
            summary[track] = {
                "success_rate": float(np.mean([o["is_success"] for o in outcomes])),
                "mean_return": float(np.mean([o["return"] for o in outcomes])),
                "mean_deviation_cm": float(np.mean([o["mean_deviation_cm"] for o in outcomes])),
                "max_deviation_cm": float(max(o["max_deviation_cm"] for o in outcomes)),
                "mean_progress": float(np.mean([o["progress_fraction"] for o in outcomes])),
                "episodes": outcomes,
            }
```

## See also

- [rover_mujoco/](../../rover_mujoco/) — the simulation environment this example depends on
- [rover_mujoco/docs/ppo-line-follower.md](../../rover_mujoco/docs/ppo-line-follower.md) — the full observation/action/reward contract and measured training results
- [Back to README](../../README.md)
