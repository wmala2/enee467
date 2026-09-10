"""Check the PPO sensor contract, outcomes, and SB3 model round trip."""

import envs  # noqa: F401
from envs.camera import line_features
from envs.pid import centroid_error
from envs.pid import LinePID
from envs.tasks.line_follower_ppo_env import LineFollowerPPOEnv
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch


def test_camera_near_feature_matches_pid():
    # A centered stripe remains distinguishable from a missing line in every band.
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert not line_features(image).any()
    image[:, 30:34] = 0
    features = line_features(image)
    assert features[0, 0] == centroid_error(image)[0]
    np.testing.assert_array_equal(features[:, 2], [1, 1, 1])
    assert features[0, 1] > features[1, 1] > features[2, 1]


def test_environment_contract_and_seeded_reset():
    environment = LineFollowerPPOEnv(max_steps=2)
    try:
        check_env(environment)
        first, _ = environment.reset(seed=41)
        second, _ = environment.reset(seed=41)
        np.testing.assert_array_equal(first, second)
        observation, _, terminated, truncated, _ = environment.step(np.zeros(2))
        np.testing.assert_array_equal(environment.data.ctrl, [-3, 3])
        assert environment.observation_space.contains(observation)
        assert np.all(observation[-2:] > 0)
        assert not terminated and not truncated
        _, _, terminated, truncated, info = environment.step(np.zeros(2))
        assert truncated and not terminated
        assert not info["is_success"]
        assert info["reason"] == "timeout"
    finally:
        environment.close()


def test_line_loss_and_off_track_are_failures():
    environment = LineFollowerPPOEnv()
    try:
        # Remove tape pixels to test genuine line loss through the renderer.
        for i in range(len(environment.points) - 1):
            environment.model.geom(f"line_{i}").rgba[:] = [0.8, 0.8, 0.8, 1]
        environment.reset(seed=0)
        for _ in range(5):
            _, _, terminated, truncated, info = environment.step(np.zeros(2))
        assert terminated and not truncated
        assert info["reason"] == "line_lost" and not info["is_success"]
        environment.reset(seed=0)
        environment.data.qpos[0] += 0.2
        _, _, terminated, truncated, info = environment.step(np.array([-1, 0]))
        assert terminated and not truncated
        assert info["reason"] == "off_track" and not info["is_success"]
    finally:
        environment.close()


def test_stopping_cannot_collect_alignment_reward():
    environment = LineFollowerPPOEnv(max_steps=3)
    try:
        environment.reset(seed=0)
        rewards = [environment.step(np.array([-1, 0]))[1] for _ in range(3)]
        assert sum(rewards) < 0
    finally:
        environment.close()


def test_pid_completes_figure_eight_through_policy_interface():
    # A full lap proves that the policy's sensors and actuator limits support both turns.
    environment = LineFollowerPPOEnv(track="figure8")
    controller = LinePID()
    try:
        observation, _ = environment.reset(seed=1000)
        while True:
            error = float(observation[0]) if observation[2] else None
            steering = controller.update(error, environment.dt) / 4
            observation, _, terminated, truncated, info = environment.step(np.array([0, steering]))
            if terminated or truncated:
                break
        assert info["is_success"], info
        assert info["max_deviation_cm"] <= 6
    finally:
        environment.close()


def test_reward_ablation_preserves_sensors_dynamics_and_outcomes():
    # Identical actions must produce identical trajectories and outcomes under both reward modes.
    traces = []
    for centering in ("camera", "chassis"):
        environment = LineFollowerPPOEnv(track="figure8", reward_centering=centering)
        observations, positions, rewards = [], [], []
        try:
            environment.reset(seed=1000)
            while True:
                observation, reward, terminated, truncated, info = environment.step(np.zeros(2))
                observations.append(observation)
                positions.append(environment.data.qpos.copy())
                rewards.append(reward)
                if terminated or truncated:
                    break
            traces.append((observations, positions, rewards, info["reason"]))
        finally:
            environment.close()
    np.testing.assert_array_equal(traces[0][0], traces[1][0])
    np.testing.assert_array_equal(traces[0][1], traces[1][1])
    assert traces[0][3] == traces[1][3] == "line_lost"
    assert not np.allclose(traces[0][2], traces[1][2])


def test_sb3_training_and_reload(tmp_path):
    # Train a tiny rollout to exercise the real Gymnasium/SB3 integration and serialization.
    torch.set_num_threads(1)
    environment = gym.make("LineFollowerPPO-v0", max_steps=20)
    try:
        model = PPO("MlpPolicy", environment, n_steps=16, batch_size=16, n_epochs=1, seed=7)
        model.learn(32)
        observation, _ = environment.reset(seed=4)
        before = model.predict(observation, deterministic=True)[0]
        model.save(tmp_path / "policy")
        loaded = PPO.load(tmp_path / "policy")
        np.testing.assert_array_equal(before, loaded.predict(observation, deterministic=True)[0])
    finally:
        environment.close()
