"""Check hardware units and enforce simulation qualification without contacting the rover."""

import hashlib
import json
import time
from unittest.mock import Mock

import numpy as np
import pytest

from rover_control.ppo_deployment import check_manifest
from rover_control.ppo_deployment import encoder_speeds
from rover_control.ppo_deployment import evaluation_evidence
from rover_control.rl_rover import RLLineFollowerRover
from rover_control.rover import Rover


def test_encoder_conversion_uses_elapsed_time_and_calibrated_signs():
    # A missed poll must preserve speed when counts and the measurement interval both double.
    np.testing.assert_allclose(encoder_speeds([100, -100], 0.1, 1000, [1, -1]), [2 * np.pi] * 2)
    np.testing.assert_allclose(encoder_speeds([200, -200], 0.2, 1000, [1, -1]), [2 * np.pi] * 2)
    with pytest.raises(ValueError):
        encoder_speeds([1, 2], 0, 1000, [1, 1])


def test_deployment_rejects_wrong_checkpoint_missing_tracks_and_low_rates(tmp_path):
    # A manifest must qualify this checkpoint in both BAM modes and the nominal baseline separately.
    model = tmp_path / "model.zip"
    model.write_bytes(b"model identity fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    evidence = {
        "model_sha256": digest,
        "rates": dict.fromkeys(("circle", "figure8", "oval"), 1.0),
        "episodes": dict.fromkeys(("circle", "figure8", "oval"), 20),
    }
    manifest = {
        "model_api": 1,
        "model_sha256": digest,
        "evaluations": {
            mode: json.loads(json.dumps(evidence)) for mode in ("nominal", "bam", "dr")
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    check_manifest(model, path)
    manifest["evaluations"]["dr"]["rates"]["figure8"] = 0.8
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dr/figure8"):
        check_manifest(model, path)
    del manifest["evaluations"]["dr"]["rates"]["figure8"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dr/figure8"):
        check_manifest(model, path)
    model.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="does not match"):
        check_manifest(model, path)


def test_evidence_recomputes_completion_instead_of_trusting_summary(tmp_path):
    # Inflated aggregate success cannot conceal failed geometric outcomes in the saved episodes.
    model = tmp_path / "model.zip"
    model.write_bytes(b"fixture")
    metadata = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "dynamics": "dr"}
    (tmp_path / "evaluation_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    episodes = [
        {
            "seed": seed,
            "dynamics": "dr",
            "is_success": True,
            "reason": "off_track",
            "max_deviation_cm": 7,
        }
        for seed in range(20)
    ]
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"circle": {"success_rate": 1, "episodes": episodes}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="below 90%"):
        evaluation_evidence(tmp_path, model, "dr")


def test_hardware_loop_stops_on_inference_or_sensor_failure(monkeypatch):
    # Bypass construction so no camera, encoder socket, or motion command reaches hardware.
    rover = object.__new__(RLLineFollowerRover)
    rover.stop = Mock()
    monkeypatch.setattr(Rover, "update", Mock(side_effect=RuntimeError("stale")))
    with pytest.raises(RuntimeError, match="stale"):
        rover.update()
    rover.stop.assert_called_once()


def test_stale_sensors_are_rejected_without_policy_inference():
    # Expired timestamps must end the loop rather than reuse an old image or encoder delta.
    rover = object.__new__(RLLineFollowerRover)
    rover._camera = Mock()
    rover._camera.latest.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), 0)
    rover._encoders = Mock()
    rover._encoders.latest.return_value = (np.zeros(2), 0.1, 0)
    rover._started = 0
    with pytest.raises(RuntimeError, match="stale"):
        rover._get_obs()


def test_runtime_uses_feature_observations_and_zeroes_subthreshold_wheel_commands():
    # Exercise the actual runtime conversion path with fresh sensor fixtures and no connections.
    rover = object.__new__(RLLineFollowerRover)
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[:, 30:34] = 0
    stamp = time.perf_counter()
    rover._camera = Mock()
    rover._camera.latest.return_value = (image, stamp)
    rover._encoders = Mock()
    rover._encoders.latest.return_value = (np.array([100, -100]), 0.1, stamp)
    rover._counts_per_revolution = 1000
    rover._encoder_signs = [1, -1]
    rover._started = stamp
    rover._lost_steps = 0
    rover.show_camera = False
    rover.wheel_diameter = 2 * 0.03435
    rover.model = Mock()
    rover.model.predict.return_value = (np.array([0, 1]), None)
    np.testing.assert_allclose(rover.compute_wheel_speeds(), [0, 7])
    observation = rover.model.predict.call_args.args[0]
    assert observation.shape == (11,) and observation[2] == 1
    np.testing.assert_allclose(observation[-2:], [2 * np.pi / 10] * 2)
