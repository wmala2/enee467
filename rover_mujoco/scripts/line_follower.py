"""Line follower: picks one of the black line tracks at random, spawns the rover at its
start point facing along the track, then steers using the rover's onboard camera (see
rover.xml's "top_cam") instead of ground-truth position. Supports both a PID controller and
a simpler bang-bang controller (see CONTROLLER below) on the same centroid-error signal.
"""

import math
import os
import random
import time

from envs.camera import set_camera_tilt
from gen_track import oval_waypoints
from gen_track import s_curve_waypoints
import mujoco
import mujoco.viewer
import numpy as np

# Each track maps to the waypoint function that generated it, so the start pose below
# stays in sync with the actual line geometry without duplicating the path math.
TRACKS = {
    "../assets/robots/rover/rover_line_oval.xml": oval_waypoints,
    "../assets/robots/rover/rover_line_s_curve.xml": s_curve_waypoints,
}

CONTROLLER = "pid"  # "pid" or "bang_bang" — see pid_control()/bang_bang_control() below

LINEAR_SPEED = 3.0  # constant forward wheel speed (rad/s)
KP, KI, KD = 6.0, 0.0, 1.5  # PID gains on normalized camera centroid error
BANG_TURN_SPEED = 3.0  # angular speed (rad/s) the bang-bang controller applies when off-line
BANG_DEADBAND = 0.05  # normalized error within which bang-bang just drives straight

# Camera tilt in degrees, 0 (straight down) to 90 (forward-facing, perpendicular to the
# rover) — see docs/pid-line-follower.md. The camera sits above the caster, so a shallow tilt
# points it at the rover's own chassis rather than the track: measured over 12 poses on each
# track, line_error() finds the line 0/12 times at 45 deg, 9/12 at 55, and 12/12 at 60.
CAMERA_ANGLE_DEG = 60.0

CAM_RES = 64  # onboard camera resolution (small + square keeps centroid math cheap)
DARK_THRESHOLD = 60  # pixel value below which we call a pixel "line"


def start_pose(waypoints_fn):
    """Rover spawn position/orientation: at the track's first waypoint, yawed to face
    the second one (so the camera starts roughly over the line, not off to one side)."""
    (x0, y0), (x1, y1) = waypoints_fn()[:2]
    tx, ty = x1 - x0, y1 - y0
    norm = math.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    # Robot "forward" is its local -Y axis -- the caster end (see teleop_rover.py's mixing);
    # solve for the yaw that rotates local -Y onto the world tangent direction (tx, ty).
    yaw = math.atan2(tx, -ty)
    quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return (x0, y0), quat


def line_error(cam_renderer, data):
    """Normalized horizontal offset of the black line's centroid from image center,
    in [-1, 1], or None if no line pixels are visible this frame."""
    cam_renderer.update_scene(data, camera="top_cam")
    img = cam_renderer.render()
    dark_cols = np.where(np.all(img < DARK_THRESHOLD, axis=-1).any(axis=0))[0]
    if len(dark_cols) == 0:
        return None
    centroid = dark_cols.mean()
    return (centroid - CAM_RES / 2) / (CAM_RES / 2)


def pid_control(error, integral, prev_error):
    """Classic PID on the centroid error. Returns (angular_speed, new_integral)."""
    integral += error
    derivative = error - prev_error
    # Negative sign: a line to the camera's right (positive error) should turn the rover
    # right (negative angular, see teleop_rover.py's turn convention).
    angular = -(KP * error + KI * integral + KD * derivative)
    return angular, integral


def bang_bang_control(error):
    """Simplest possible line follower: full turn one way or the other, nothing
    proportional. A small deadband around zero keeps it from chattering when centered."""
    if error > BANG_DEADBAND:
        return -BANG_TURN_SPEED
    if error < -BANG_DEADBAND:
        return BANG_TURN_SPEED
    return 0.0


def main():
    scene_rel, wp_fn = random.choice(list(TRACKS.items()))
    model_path = os.path.join(os.path.dirname(__file__), scene_rel)
    print(f"Track: {os.path.basename(scene_rel)}, controller: {CONTROLLER}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    set_camera_tilt(model, "top_cam", CAMERA_ANGLE_DEG)

    (sx, sy), quat = start_pose(wp_fn)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1], data.qpos[2] = sx, sy, 0.1
    data.qpos[3:7] = quat
    mujoco.mj_forward(model, data)

    cam_renderer = mujoco.Renderer(model, height=CAM_RES, width=CAM_RES)
    integral = 0.0
    prev_error = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Track geoms are group 3; the viewer shows only 0-2 by default.
        viewer.opt.geomgroup[3] = 1
        while viewer.is_running():
            step_start = time.time()

            error = line_error(cam_renderer, data)
            if error is None:
                error = prev_error  # line briefly out of view: hold last correction

            if CONTROLLER == "bang_bang":
                angular = bang_bang_control(error)
            else:
                angular, integral = pid_control(error, integral, prev_error)
            prev_error = error

            # Forward is [-v, +v], the same mixing teleop_rover.py uses. This was inverted
            # here, which drove the rover caster-trailing -- backwards, by the model's own
            # naming -- and every script derived from this one inherited it.
            data.ctrl[:] = [-LINEAR_SPEED + angular, LINEAR_SPEED + angular]

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
