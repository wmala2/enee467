"""Runtime control of the rover's onboard camera tilt, mirroring the physical rover's
adjustable camera mount (0 deg = straight down, 90 deg = forward-facing/perpendicular to the
rover, per docs/pid-line-follower.md). Mutating a compiled model's cam_quat like this is the
same technique LineFollowerEnv's domain randomization uses to jitter the camera each reset.
"""

import math

import mujoco


def set_camera_tilt(model, cam_name, angle_deg):
    """Point `cam_name` `angle_deg` up from straight-down (0) toward forward-facing (90),
    rotating about the camera mount's own local X (its lateral/hinge axis).

    The quaternion is a 180-degree yaw composed with the tilt, not the tilt alone, and all
    three of the camera's axes depend on getting that right. MuJoCo looks along -Z_cam with
    +Y_cam up and +X_cam right in the image; this rover's forward is local -Y (see
    teleop_rover.py's mixing).

    Tilting about +X alone points the view at the rover's rear. Tilting about -X points it
    forward but leaves the frame rolled 180 degrees -- upside down *and* left-right mirrored,
    so the line-centering error came out negated and the picture showed sky where the ground
    should be. Composing a 180-degree yaw with the tilt gives view -Y (forward), up +Z, and
    image-right along body -X, which is the rover's right."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    theta = math.radians(angle_deg)
    model.cam_quat[cam_id] = [0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)]


# Geom groups the rover's own camera has to be told to draw. MuJoCo's default MjvOption is
# geomgroup = [1, 1, 1, 0, 0, 0], and mujoco.Renderer.update_scene() uses that default unless
# a scene_option is passed. The line tracks are group 3 and the goal-nav obstacles are group 4,
# so every onboard render in this repo silently excluded the one thing the policy needed to
# see: the camera returned floor and shadows, never the track or an obstacle.
ONBOARD_GEOM_GROUPS = (3, 4)


def onboard_scene_option():
    """Scene options for any render that stands in for the rover's own camera.

    Always pass this to Renderer.update_scene(). Without it the track and the obstacles are
    invisible to the policy -- which does not fail loudly, it just produces a camera that
    always shows an empty floor, so anything trained on it learns to ignore the image."""
    option = mujoco.MjvOption()
    for group in ONBOARD_GEOM_GROUPS:
        option.geomgroup[group] = 1
    return option
