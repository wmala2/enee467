"""Line follower: picks one of the black line tracks at random, spawns the rover at its
start point facing along the track, then steers using the rover's onboard camera (see
rover.xml's "top_cam") instead of ground-truth position. Supports both a PID controller and
a simpler bang-bang controller (see CONTROLLER below) on the same centroid-error signal.

    uv run python scripts/line_follower.py                 # random track, MuJoCo viewer
    uv run python scripts/line_follower.py --track circle  # a specific track
    uv run python scripts/line_follower.py --list-tracks   # what is available
    uv run python scripts/line_follower.py --camera        # live onboard-camera window
    uv run python scripts/line_follower.py --controller bang_bang

The --camera window is what the controller actually sees: the 64x64 frame, the near-field band
the centroid is taken over, the detected line pixels, and the resulting error. Tuning KP/KI/KD
by watching the rover from outside is guesswork; tuning them against this is not.
"""

import argparse
import math
import os
import random
import time

from envs.camera import onboard_scene_option
from envs.camera import set_camera_tilt
from gen_track import oval_waypoints
from gen_track import s_curve_waypoints
import mujoco
import mujoco.viewer
import numpy as np

# Generated tracks: each maps to the waypoint function that produced it, so the start pose
# stays in sync with the actual line geometry without duplicating the path math.
TRACKS = {
    "../assets/robots/rover/rover_line_oval.xml": oval_waypoints,
    "../assets/robots/rover/rover_line_s_curve.xml": s_curve_waypoints,
}

# Imported tracks: real geometry converted from the Webots PROTOs by
# scripts/import_webots_track.py. These are meshes rather than parametric curves, so there is
# no waypoint function to spawn from -- mesh_start_pose() derives a pose from the mesh itself.
IMPORTED_TRACKS = {
    "figure8": ("../assets/robots/rover/rover_Figure8Track.xml", "Figure8Track"),
}

# The other six Webots tracks convert and load but do not render, so the camera cannot see
# them and the follower drives blind. They are listed separately rather than offered and left
# to fail.
#
# Cause: the OnShape exporter writes each track as a *solid* whose thickness rounds to zero --
# every vertex lands on the same z, extent exactly 0.000 -- so its side walls are zero-area
# triangles that MuJoCo discards at compile. CircleTrack's 1128 faces become 288, MidTrack's
# 996 become 2, and what survives does not form a visible surface. Figure8Track came out of
# Tinkercad instead and has real geometry, which is what isolates the exporter rather than the
# importer: normalizing the face format (f v//vn -> f v v v) was necessary but not sufficient.
#
# The fix is not more mesh wrangling. Extracting each track's centreline and emitting it as the
# same thin-box polyline scripts/gen_track.py already produces would render reliably, collide
# correctly, and -- the part that matters for RL -- yield the waypoints LineFollowerEnv needs
# for its progress reward, which a mesh cannot provide.
UNSUPPORTED_TRACKS = {
    "circle": "CircleTrack",
    "goomba": "GoombaTrack",
    "hard": "HardTrack",
    "medium": "MediumCompTrack",
    "mid": "MidTrack",
    "swing": "SwingTrack",
}
GENERATED_NAMES = {"oval": 0, "s_curve": 1}

CONTROLLER = "pid"  # "pid" or "bang_bang" — see pid_control()/bang_bang_control() below

LINEAR_SPEED = 3.0  # constant forward wheel speed (rad/s)
# PID gains on the normalized camera centroid error. Swept against the working error signal
# (the old 6.0/0.0/1.5 were tuned when the camera could not see the track at all, so they were
# fitted to noise). Lower KP tracks better here: the plant is already well damped by the motor
# model, and the error signal is a look-ahead measurement, which adds its own phase lead.
#
# KD earns very little on this track -- 1.5 and 3.0 give identical trajectories, because the
# error changes slowly compared to the 10 Hz control step. It is kept small and nonzero rather
# than removed, since a faster LINEAR_SPEED or a tighter track would give it something to do.
KP, KI, KD = 2.0, 0.0, 1.0
BANG_TURN_SPEED = 3.0  # angular speed (rad/s) the bang-bang controller applies when off-line
BANG_DEADBAND = 0.05  # normalized error within which bang-bang just drives straight

# Camera tilt in degrees, 0 (straight down) to 90 (forward-facing, perpendicular to the
# rover) — see docs/pid-line-follower.md. The camera sits above the caster, so a shallow tilt
# points it at the rover's own chassis rather than the track: measured over 12 poses on each
# track, line_error() finds the line 0/12 times at 45 deg, 9/12 at 55, and 12/12 at 60.
CAMERA_ANGLE_DEG = 60.0

CAM_RES = 64  # onboard camera resolution (small + square keeps centroid math cheap)
DARK_THRESHOLD = 60  # pixel value below which we call a pixel "line"
# Fraction of the frame, measured up from the bottom, that the centroid is taken over. The
# near ground: the only part of a forward-tilted view where "dark" can only mean track.
#
# This is the single biggest lever on tracking accuracy, far more than any gain. A wide band
# samples the line further ahead, and steering to centre a look-ahead point puts the *body*
# outside the curve -- measured, the offset at which the error reads zero swings from -13.5 cm
# to +14.2 cm depending on local curvature, which no gain can correct because it is geometry
# rather than dynamics. Narrowing the band samples closer to the wheels:
#     band  mean deviation  max
#     0.50      5.3 cm      10.1 cm
#     0.30      4.5 cm       8.7 cm
#     0.20      4.0 cm       7.9 cm
#     0.15      3.4 cm       6.8 cm
# Below ~0.12 the band gets too few pixels to be reliable on a faint or distant line.
GROUND_BAND = 0.15


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


def mesh_start_pose(model, geom_name, rng):
    """A spawn pose on an imported mesh track.

    The generated tracks come with waypoints; a mesh does not, so the pose is derived from the
    geometry: pick a vertex, fit the local tangent to its neighbours, and choose the direction
    along that tangent with more track ahead of it. Fitting a direction to neighbours gives an
    axis but not a sign, and picking the wrong sign spawns the rover facing off the end of the
    line."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[gid]
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[start : start + count].astype(np.float64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
    world = verts @ rot.reshape(3, 3).T + model.geom_pos[gid]
    xy = world[:, :2]

    origin = xy[rng.integers(len(xy))]
    neighbours = xy[np.argsort(np.linalg.norm(xy - origin, axis=1))[1:12]]
    centred = neighbours - neighbours.mean(axis=0)
    # Principal direction of the neighbourhood is the line's local tangent.
    tangent = np.linalg.svd(centred, full_matrices=False)[2][0]
    ahead = [
        np.sum(np.linalg.norm(xy - (origin + sign * tangent * 0.15), axis=1) < 0.12)
        for sign in (1.0, -1.0)
    ]
    tangent = tangent * (1.0 if ahead[0] >= ahead[1] else -1.0)

    yaw = math.atan2(tangent[0], -tangent[1])  # local -Y onto the tangent
    return (float(origin[0]), float(origin[1])), [
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    ]


def line_error(cam_renderer, data):
    """Normalized horizontal offset of the black line's centroid from image center,
    in [-1, 1], or None if no line pixels are visible this frame.

    Only the bottom GROUND_BAND of the frame is used. That is standard line-follower practice
    -- it is the ground just ahead of the rover, which is what you actually steer on -- and
    here it is also load-bearing: with a forward-tilted camera the upper frame contains the
    horizon, and anything above it is sky rather than floor. Measuring the whole frame let
    distant track (and, before the scenes gained a skybox, the black void itself) drag the
    centroid toward the middle, which pinned the error near zero and drove the rover straight
    past every bend.

    The centroid is weighted by how many dark pixels each column holds, not by which columns
    happen to contain one, so a thick near stripe outvotes a thin far one."""
    cam_renderer.update_scene(data, camera="top_cam", scene_option=onboard_scene_option())
    img = cam_renderer.render()
    band = img[int(CAM_RES * (1.0 - GROUND_BAND)) :, :, :]
    dark = np.all(band < DARK_THRESHOLD, axis=-1)
    weights = dark.sum(axis=0).astype(np.float64)
    if weights.sum() == 0:
        return None
    centroid = float((np.arange(CAM_RES) * weights).sum() / weights.sum())
    return (centroid - CAM_RES / 2) / (CAM_RES / 2)


def camera_overlay(img, error, scale=6):
    """The onboard frame, blown up, with the near-field band and detected line marked.

    Draws with numpy rather than cv2 so this stays a MuJoCo-only dependency; the window itself
    is cv2, which the workspace already ships for the ArUco and YOLO packages."""
    view = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1).astype(np.uint8).copy()
    view = np.vstack([np.zeros((26, view.shape[1], 3), dtype=np.uint8), view])
    band_top = int(CAM_RES * (1.0 - GROUND_BAND)) * scale
    view[band_top, :, :] = [0, 160, 255]  # boundary of the band the centroid uses

    dark = np.all(img < DARK_THRESHOLD, axis=-1)
    dark_big = np.repeat(np.repeat(dark, scale, axis=0), scale, axis=1)
    tint = np.zeros_like(view)
    tint[..., 0] = 255
    view[dark_big] = (0.45 * view[dark_big] + 0.55 * tint[dark_big]).astype(np.uint8)

    mid = view.shape[1] // 2
    view[26:, mid - 1 : mid + 1, :] = [90, 90, 90]  # image centre
    if error is not None:
        col = int((error * (CAM_RES / 2) + CAM_RES / 2) * scale)
        col = max(1, min(view.shape[1] - 2, col))
        view[26:, col - 1 : col + 1, :] = [0, 255, 0]  # measured line centroid
    return view


def read_telemetry(model, data):
    """(forward speed m/s, left wheel rad/s, right wheel rad/s) from the model's own sensors.

    Read through sensordata rather than qvel so this reports exactly what the viewer's sensor
    plot is drawing, and exactly what a real rover would publish."""
    adr = {}
    for name in ("base_velocity", "left_wheel_speed", "right_wheel_speed"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr[name] = model.sensor_adr[sid]
    vel = data.sensordata[adr["base_velocity"] : adr["base_velocity"] + 3]
    # The velocimeter is in the site's frame, and the rover's forward is its local -Y.
    return (
        -float(vel[1]),
        float(data.sensordata[adr["left_wheel_speed"]]),
        float(data.sensordata[adr["right_wheel_speed"]]),
    )


def show_telemetry(viewer, data, speed, error):
    """Float a text readout above the rover in the MuJoCo viewer.

    user_scn is the viewer's scratch scene for caller-supplied geoms. A fully transparent
    sphere carries the label without drawing anything, which is the least intrusive way to get
    text into the 3D view -- the passive viewer has no text overlay API of its own."""
    scene = viewer.user_scn
    scene.ngeom = 0
    marker = scene.geoms[0]
    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.005, 0.0, 0.0]),
        data.qpos[:3] + np.array([0.0, 0.0, 0.22]),
        np.eye(3).flatten(),
        np.array([1.0, 1.0, 1.0, 0.0]),
    )
    err = "  line lost" if error is None else f"  err {error:+.2f}"
    marker.label = f"{speed:+.2f} m/s{err}"
    scene.ngeom = 1


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", action="store_true", help="show a live window of the onboard camera"
    )
    parser.add_argument(
        "--track",
        help="which track to drive; omit for a random generated one, --list-tracks to see all",
    )
    parser.add_argument("--list-tracks", action="store_true", help="list tracks and exit")
    parser.add_argument(
        "--controller", choices=("pid", "bang_bang"), default=CONTROLLER, help="steering law"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for imported-track spawn")
    args = parser.parse_args()

    if args.list_tracks:
        print("generated (parametric, with waypoints):")
        for name in GENERATED_NAMES:
            print(f"  {name}")
        print("imported (Webots meshes, spawn derived from geometry):")
        for name, (_, geom) in IMPORTED_TRACKS.items():
            print(f"  {name:10s} ({geom})")
        print("unavailable — convert and load, but do not render (see UNSUPPORTED_TRACKS):")
        for name, geom in UNSUPPORTED_TRACKS.items():
            print(f"  {name:10s} ({geom})")
        return

    controller = args.controller
    show_camera = args.camera
    if show_camera:
        import cv2

    rng = np.random.default_rng(args.seed)
    imported_geom = None
    if args.track in IMPORTED_TRACKS:
        scene_rel, imported_geom = IMPORTED_TRACKS[args.track]
        wp_fn = None
    elif args.track in GENERATED_NAMES:
        scene_rel = list(TRACKS)[GENERATED_NAMES[args.track]]
        wp_fn = TRACKS[scene_rel]
    elif args.track is None:
        scene_rel, wp_fn = random.choice(list(TRACKS.items()))
    elif args.track in UNSUPPORTED_TRACKS:
        raise SystemExit(
            f"{args.track!r} converts but does not render — its OnShape export collapsed to a "
            "zero-thickness solid, so MuJoCo discards its faces. See UNSUPPORTED_TRACKS."
        )
    else:
        known = ", ".join(list(GENERATED_NAMES) + list(IMPORTED_TRACKS))
        raise SystemExit(f"unknown track {args.track!r}; choose from: {known}")

    model_path = os.path.join(os.path.dirname(__file__), scene_rel)
    print(f"Track: {os.path.basename(scene_rel)}, controller: {controller}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    set_camera_tilt(model, "top_cam", CAMERA_ANGLE_DEG)

    if imported_geom is not None:
        (sx, sy), quat = mesh_start_pose(model, imported_geom, rng)
    else:
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
            speed, left_rad_s, right_rad_s = read_telemetry(model, data)
            show_telemetry(viewer, data, speed, error)
            if show_camera:
                cam_renderer.update_scene(
                    data, camera="top_cam", scene_option=onboard_scene_option()
                )
                frame = camera_overlay(cam_renderer.render(), error)
                readout = f"{speed:+.2f} m/s   L {left_rad_s:+5.2f}  R {right_rad_s:+5.2f}   " + (
                    "line lost" if error is None else f"err {error:+.3f}"
                )
                cv2.putText(
                    frame, readout, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
                )
                cv2.imshow("rover camera (red = line, green = centroid)", frame[:, :, ::-1])
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if error is None:
                error = prev_error  # line briefly out of view: hold last correction

            if controller == "bang_bang":
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
