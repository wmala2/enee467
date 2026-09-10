"""Exercise DR reproducibility, physical effects, and the policy interface in MuJoCo."""

from envs.contract import observation_from_sensors
from envs.contract import wheel_targets
from envs.line_dynamics import NOMINAL
from envs.line_dynamics import validate_ranges
from envs.tasks.line_follower_ppo_env import LineFollowerPPOEnv
import numpy as np
import pytest


def test_dr_replays_physics_pixels_and_parameters_without_reset_drift():
    # Replay a full noisy trace after an intervening seed to catch lingering model or RNG state.
    environment = LineFollowerPPOEnv(dynamics="dr", max_steps=8)
    assert environment.dynamics is not None
    traces = []
    try:
        for seed in (41, 42, 41):
            observation, info = environment.reset(seed=seed)
            trace = [observation]
            for _ in range(8):
                observation, _, terminated, truncated, _ = environment.step(np.zeros(2))
                trace.append(np.concatenate((observation, environment.data.qpos.copy())))
                if terminated or truncated:
                    break
            traces.append((info, trace))
            assert environment.observation_space.contains(observation)
            np.testing.assert_allclose(
                environment.model.body_mass,
                environment.dynamics.mass * info["domain_parameters"]["mass_scale"],
            )
            assert all(
                contact.friction[0] == pytest.approx(info["domain_parameters"]["contact_friction"])
                for contact in environment.data.contact
            )
        assert traces[0][0] == traces[2][0]
        assert traces[0][0] != traces[1][0]
        for first, repeated in zip(traces[0][1], traces[2][1], strict=True):
            np.testing.assert_array_equal(first, repeated)
    finally:
        environment.close()


def test_delayed_command_and_motor_torque_actually_reach_physics():
    # Fixed parameter bounds isolate one control-step delay from other random effects.
    ranges = {key: [value, value] for key, value in NOMINAL.items()}
    ranges["command_delay_steps"] = [1, 1]
    environment = LineFollowerPPOEnv(dynamics="dr", dr_ranges=ranges, max_steps=4)
    assert environment.dynamics is not None
    try:
        environment.reset(seed=10)
        environment.step(np.zeros(2))
        np.testing.assert_allclose(environment.data.ctrl, 0, atol=1e-12)
        environment.step(np.zeros(2))
        assert environment.data.ctrl[0] < 0 < environment.data.ctrl[1]
        assert np.all(environment.wheel_speeds > 1)
        assert np.all(environment.model.dof_damping[environment.dynamics.dofs] > 0)
        np.testing.assert_allclose(environment.model.actuator_gainprm[:, 0], 1)
        np.testing.assert_array_equal(environment.model.actuator_biasprm, 0)
    finally:
        environment.close()


def test_shared_interface_preserves_forward_and_turn_signs():
    # Physical wheel signs are positive forward while only MuJoCo negates the left axle.
    np.testing.assert_array_equal(wheel_targets([0, 0]), [3, 3])
    np.testing.assert_array_equal(wheel_targets([0, 1]), [-1, 7])
    np.testing.assert_array_equal(wheel_targets([-1, 0]), [0, 0])
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[:, 30:34] = 0
    observation = observation_from_sensors(image, [3, 4])
    assert observation.shape == (11,) and observation.dtype == np.float32
    np.testing.assert_allclose(observation[-2:], [0.3, 0.4])
    with pytest.raises(ValueError, match="finite"):
        wheel_targets([np.nan, 0])


@pytest.mark.parametrize(
    "override",
    [
        {"voltage": [-1, 12]},
        {"typo": [0, 1]},
        {"mass_scale": [2, 1]},
        {"command_delay_steps": [0, 0.5]},
    ],
)
def test_invalid_ranges_fail_before_training(override):
    # Invalid physical parameters must fail before worker processes begin collecting rollouts.
    with pytest.raises(ValueError):
        validate_ranges(override)
