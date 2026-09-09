"""Three invariants adopted from the Microduck sim2real playbook.

Each of these exists because breaking it produces a run that looks healthy while training
something other than the task. They are cheap, they run on CPU, and they are meant to be fast
enough that there is never a reason to skip them before a long run.
"""

import subprocess
import sys

from envs.tasks.line_follower_env import LineFollowerEnv
from envs.tasks.line_follower_ppo_env import LineFollowerPPOEnv
from envs.tasks.line_follower_ppo_env import TRACKS
import numpy as np


def test_penalty_terms_are_never_positive():
    """Every reported penalty must be <= 0, on every step.

    A penalty whose sign is flipped stops being a cost and becomes a bounty for exactly the
    behavior it was written to suppress, and PPO finds that long before a human rereads the
    reward code. The sequence below is deliberately ugly, hard alternating steering, because
    that is what the smoothness term exists to charge for: if its sign is ever wrong, this is
    the input that pays out.
    """
    environment = LineFollowerPPOEnv(track="figure8")
    try:
        environment.reset(seed=3)
        for step in range(150):
            action = np.array([0.5, 1.0 if step % 2 else -1.0], dtype=np.float32)
            _, _, terminated, truncated, info = environment.step(action)
            assert info["reward_smoothness"] <= 0, (
                f"reward_smoothness was {info['reward_smoothness']} at step {step}"
            )
            if terminated or truncated:
                environment.reset(seed=3)
    finally:
        environment.close()


def test_domain_randomization_does_not_accumulate():
    """Randomized quantities must be re-derived from stored defaults, never compounded.

    Randomization that applies each episode's jitter on top of the previous episode's value
    walks away from the nominal model a little further every reset, so a long run spends most
    of its samples on a scene that no longer resembles the robot. The failure is silent for
    hours, since early episodes look right and the only symptom is a policy that will not
    transfer. Returning to a seed after many intervening resets has to reproduce the same
    scene exactly; under an accumulating randomizer it cannot.
    """
    environment = LineFollowerEnv(domain_randomize=True)
    try:
        environment.reset(seed=11)
        first_camera = environment.model.cam_pos[environment._cam_id].copy()
        first_light = environment.model.light_diffuse.copy()
        for seed in range(40):
            environment.reset(seed=seed)
        environment.reset(seed=11)
        np.testing.assert_allclose(
            environment.model.cam_pos[environment._cam_id],
            first_camera,
            err_msg="camera pose drifted across resets; randomization is accumulating",
        )
        np.testing.assert_allclose(
            environment.model.light_diffuse,
            first_light,
            err_msg="light intensity drifted across resets; randomization is accumulating",
        )
    finally:
        environment.close()


def test_training_entry_point_smoke_runs(tmp_path):
    """Start the real training CLI and let it finish a tiny run.

    The rest of the suite exercises the environment and the config; nothing else actually
    enters the training loop, which is where argument plumbing, vector-env construction,
    logging and checkpoint writing live. A handful of steps is enough to reach all of them,
    and this is the check to run before spending an hour on a real budget.
    """
    output = tmp_path / "smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_ppo.py",
            "--timesteps",
            "128",
            # One worker per track: the CLI requires every track to be represented.
            "--n-envs",
            str(len(TRACKS)),
            "--output",
            str(output),
            "--no-wandb",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, (
        f"training CLI failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )
    assert (output / "config.json").exists(), "run configuration was not written"
    assert (output / "final_model.zip").exists(), "final policy was not saved"
