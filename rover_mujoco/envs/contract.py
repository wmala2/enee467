"""The deployment contract: what a published policy must look like to run on the rover.

Kept in one place because both sides depend on it. The environment produces observations of
this shape and the runtime in rover_control/rl_rover.py must build the same thing; a mismatch
does not fail loudly at the boundary, it fails on hardware.
"""

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
