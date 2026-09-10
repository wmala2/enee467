"""The deployment contract: what a published policy must look like to run on the rover.

Kept in one place because both sides depend on it. The environment produces observations of
this shape and the runtime in rover_control/rl_rover.py must build the same thing; a mismatch
does not fail loudly at the boundary, it fails on hardware.
"""

import numpy as np

from envs.camera import line_features

SCHEMA_VERSION = 1
# Bump when the observation or action layout changes. A runtime that knows only an older API
# must refuse a newer policy rather than feed it a wrongly shaped array.
MODEL_API = 1

OBS_LEN = 11  # 3 bands x (centroid x, centroid y, visible) + 2 wheel speeds
ACTION_LEN = 2  # [forward, steering], both in [-1, 1]
CONTROL_HZ = 10.0  # must match Rover.COMMAND_RATE_HZ on the hardware side

OBS_LAYOUT = (
    "0-2 near band (x, y, visible); 3-5 middle band; 6-8 far band; "
    "9-10 left/right wheel speed / 10 rad/s"
)
ACTION_LAYOUT = "0 forward in [-1, 1] -> 0..6 rad/s; 1 steering in [-1, 1] -> +/-4 rad/s"


def wheel_targets(action):
    """Decode normalized actions to physical left/right rad/s, positive forward on both wheels."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_LEN,) or not np.isfinite(action).all():
        raise ValueError("action must contain two finite normalized commands")
    # MuJoCo alone negates the left target because the CAD axle uses the opposite axis.
    forward, steering = np.clip(action, -1, 1) * [3.0, 4.0] + [3.0, 0.0]
    return np.array([forward - steering, forward + steering])


def observation_from_sensors(rgb, wheel_rad_s):
    """Build the same eleven policy inputs from simulation or physical sensor measurements."""
    speeds = np.asarray(wheel_rad_s, dtype=float)
    if speeds.shape != (2,) or not np.isfinite(speeds).all():
        raise ValueError("wheel speeds must contain two finite values in rad/s")
    # Preserve the PID's threshold and camera bands across training and deployment.
    return np.concatenate((line_features(rgb).flatten(), np.clip(speeds / 10, -1, 1))).astype(
        np.float32
    )
