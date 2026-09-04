"""Runtime control of the rover's onboard camera tilt, mirroring the physical rover's
adjustable camera mount (0 deg = straight down, 90 deg = forward-facing/perpendicular to the
rover, per docs/pid-line-follower.md). Mutating a compiled model's cam_quat like this is the
same technique LineFollowerEnv's domain randomization uses to jitter the camera each reset.
"""

import math

import mujoco


def set_camera_tilt(model, cam_name, angle_deg):
    """Point `cam_name` `angle_deg` up from straight-down (0) toward forward-facing (90),
    rotating about the camera mount's own local X (its lateral/hinge axis)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    theta = math.radians(angle_deg)
    model.cam_quat[cam_id] = [math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0]
