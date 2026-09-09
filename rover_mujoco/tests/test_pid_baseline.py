"""Regression checks for the camera controller and ordered traversal evaluation."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# Make the standalone runner importable with the same sibling imports as its CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from envs.pid import centroid_error
from envs.pid import LinePID
from envs.tracks import figure8_waypoints
from envs.tracks import path_length_table
from envs.tracks import project_arc_length
from scripts.line_follower import build_model
from scripts.line_follower import camera_overlay
from scripts.line_follower import run_episode


def test_integral_uses_elapsed_seconds():
    # A constant error produces the same integral at two different sampling rates.
    slow, fast = LinePID(0, 1, 0), LinePID(0, 1, 0)
    for _ in range(10):
        slow.update(0.5, 0.1)
    for _ in range(100):
        fast.update(0.5, 0.01)
    assert slow.integral == pytest.approx(fast.integral)


def test_saturation_and_dropout_reset():
    # Sustained saturation must not accumulate integral or kick on reacquisition.
    pid = LinePID(2, 1, 1, limit=1)
    for _ in range(100):
        assert pid.update(1, 0.1) == -1
    assert pid.integral == 0
    assert pid.update(None, 0.1) == 0
    assert pid.update(-0.1, 0.1) == pytest.approx(0.21)
    with pytest.raises(ValueError):
        pid.update(0, 0)


def test_camera_distinguishes_centered_line_from_missing_line():
    # Symmetric pixels yield zero error, whereas a blank image explicitly yields None.
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert centroid_error(image)[0] is None
    image[62, 29] = 0
    assert centroid_error(image)[0] is None
    image[:] = 255
    image[55:, 29:35] = 0
    error, mask = centroid_error(image)
    assert error == pytest.approx(0)
    assert camera_overlay(image, error, mask).shape == (384, 384, 3)
    image[55:] = 255
    image[55:, 45:49] = 0
    error, _ = centroid_error(image)
    assert error > 0
    assert LinePID().update(error, 0.1) < 0


def test_figure_eight_crossing_preserves_branch():
    # The same crossing coordinate represents two distinct places along the ordered lap.
    points = np.asarray(figure8_waypoints())
    lengths, closed = path_length_table(points)
    first, _ = project_arc_length(points, lengths, (0, 0), 40, window=4, closed=closed)
    second, _ = project_arc_length(points, lengths, (0, 0), 120, window=4, closed=closed)
    assert closed
    assert second - first == pytest.approx(lengths[-1] / 2)


def test_baseline_scene_uses_two_inch_tape_and_measured_mount():
    # Compile the actual CAD scene to catch include, mesh-path, width, and camera regressions.
    model = build_model(figure8_waypoints(), 45)
    assert model.geom("line_0").size[1] * 2 == pytest.approx(0.0508)
    assert model.camera("top_cam").pos[2] == pytest.approx(-0.0243587 + 0.05)
    assert model.camera("top_cam").quat[2] == pytest.approx(np.sin(np.pi / 8))


@pytest.mark.parametrize("blank_track,expected", [(True, "line_lost"), (False, "timeout")])
def test_failed_episodes_cannot_count_as_completion(tmp_path, blank_track, expected):
    # Exercise actual camera rendering and stop conditions, including missing-line handling.
    points = figure8_waypoints()
    model = build_model(points, 45)
    if blank_track:
        for i in range(len(points) - 1):
            model.geom(f"line_{i}").rgba[:] = [0.8, 0.8, 0.8, 1]
    args = SimpleNamespace(
        kp=2,
        ki=0,
        kd=0.1,
        speed=3,
        headless=True,
        camera=False,
        controller="pid",
        duration=0.6,
        max_deviation=0.06,
    )
    outcome, _ = run_episode(model, points, args, 0, tmp_path)
    assert outcome["reason"] == expected
    assert not outcome["success"]
