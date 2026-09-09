"""Camera-only PID baseline on 2-inch circle, figure-eight, oval, and S-curve tracks.

Run from rover_mujoco: uv run scripts/line_follower.py --track figure8 --camera
Headless evaluation: MUJOCO_GL=egl uv run scripts/line_follower.py --headless --track all
"""

import argparse
from contextlib import nullcontext
import csv
import json
import math
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

from envs.pid import centroid_error
from envs.pid import LinePID
from envs.tracks import circle_waypoints
from envs.tracks import figure8_waypoints
from envs.tracks import oval_waypoints
from envs.tracks import path_length_table
from envs.tracks import project_arc_length
from envs.tracks import s_curve_waypoints
from gen_track import build_track_xml
import mujoco
import mujoco.viewer
import numpy as np

TRACKS = {
    "circle": circle_waypoints,
    "figure8": figure8_waypoints,
    "oval": oval_waypoints,
    "s_curve": s_curve_waypoints,
}
ROVER_DIR = Path(__file__).resolve().parents[1] / "assets/robots/rover"
LINE_WIDTH = 0.0508
CAM_RES = 64
CONTROL_HZ = 10


def build_model(waypoints, tilt):
    """Reuse the CAD rover and velocity-servo scene, replacing only the baseline track."""
    # Expand the two local includes in memory so existing RL scene assets stay reproducible.
    scene = ET.parse(ROVER_DIR / "rover_line_oval.xml").getroot()
    for include in list(scene.findall("include")):
        scene.remove(include)
    # MuJoCo accepts double hyphens in legacy comments, but ElementTree rejects them.
    rover_xml = (ROVER_DIR / "rover.xml").read_text(encoding="utf-8")
    rover = ET.fromstring(re.sub(r"<!--.*?-->", "", rover_xml, flags=re.DOTALL))
    for mesh in rover.findall(".//mesh"):
        mesh.set("file", str(ROVER_DIR / mesh.attrib["file"]))
    scene.extend(rover)
    scene.extend(ET.fromstring(build_track_xml("pid_track", waypoints, LINE_WIDTH)))
    # MuJoCo exports compiled bindings without type stubs; exercise them in simulation tests.
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))  # ty: ignore[unresolved-attribute]
    # Match the measured 45-degree mount height above the CAD plate; forward offset is provisional.
    camera = model.camera("top_cam")
    camera.pos[:] = [0, -0.108, -0.0243587 + 0.05]
    angle = math.radians(tilt) / 2
    camera.quat[:] = [0, 0, math.sin(angle), math.cos(angle)]
    return model


def camera_overlay(image, error, mask):
    """Enlarge the actual sensor image and highlight only the pixels used by the PID."""
    view = image.copy()
    view[mask] = [255, 60, 60]
    view[:, CAM_RES // 2] = [90, 90, 90]
    if error is not None:
        col = round((error + 1) * (CAM_RES - 1) / 2)
        view[:, col] = [0, 255, 0]
    return np.repeat(np.repeat(view, 6, axis=0), 6, axis=1)


def run_episode(model, waypoints, args, seed, output):
    """Control with pixels; use ordered path projection only for independent evaluation."""
    rng = np.random.default_rng(seed)
    points = np.asarray(waypoints)
    lengths, closed = path_length_table(points)
    tangent = points[1] - points[0]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    lateral = rng.uniform(-0.01, 0.01)
    yaw_offset = rng.uniform(-math.radians(5), math.radians(5))
    yaw = math.atan2(tangent[0], -tangent[1]) + yaw_offset
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    data.qpos[:2] = points[0] + lateral * normal
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    # Settle the free body before measuring the camera and starting the control clock.
    mujoco.mj_step(model, data, nstep=round(1 / model.opt.timestep))  # ty: ignore[unresolved-attribute]
    mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]
    substeps = round(1 / CONTROL_HZ / model.opt.timestep)
    dt = substeps * model.opt.timestep
    pid = LinePID(args.kp, args.ki, args.kd, 10 - args.speed)
    s_prev, segment = project_arc_length(points, lengths, data.qpos[:2], 0, closed=closed)
    progress = 0.0
    missing = 0
    last_steering = 0.0
    rows = []
    reason = "timeout"
    initial_camera = data.cam_xpos[model.camera("top_cam").id].copy()
    viewer_context = nullcontext() if args.headless else mujoco.viewer.launch_passive(model, data)
    # Headless and interactive modes use the same camera, controller, and physics cadence.
    with (
        mujoco.Renderer(model, height=CAM_RES, width=CAM_RES) as renderer,
        viewer_context as viewer,
    ):
        if viewer is not None:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 2.8
            viewer.cam.elevation = -75
        for step in range(math.ceil(args.duration / dt)):
            wall_start = time.monotonic()
            renderer.update_scene(data, camera="top_cam")
            frame = renderer.render()
            error, mask = centroid_error(frame)
            if step == 0:
                from PIL import Image

                Image.fromarray(camera_overlay(frame, error, mask)).save(
                    output / f"camera_{seed}.png"
                )
            missing = missing + 1 if error is None else 0
            if missing * dt >= 0.5:
                data.ctrl[:] = 0
                reason = "line_lost"
                break
            # Hold steering briefly across a dropout, resetting PID memory on reacquisition.
            correction = pid.update(error, dt)
            steering = last_steering if error is None else correction
            if args.controller == "bang_bang" and error is not None:
                steering = -3.0 * np.sign(error) if abs(error) > 0.05 else 0.0
            steering = float(np.clip(steering, -pid.limit, pid.limit))
            last_steering = steering
            data.ctrl[:] = [-args.speed + steering, args.speed + steering]
            mujoco.mj_step(model, data, nstep=substeps)  # ty: ignore[unresolved-attribute]
            mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]
            # Restrict projection to nearby segments to prevent branch jumps at the crossing.
            s, segment = project_arc_length(
                points, lengths, data.qpos[:2], segment, window=4, closed=closed
            )
            delta = s - s_prev
            if closed:
                delta = (delta + lengths[-1] / 2) % lengths[-1] - lengths[-1] / 2
            progress += delta
            s_prev = s
            fraction = (s - lengths[segment]) / (lengths[segment + 1] - lengths[segment])
            projection = points[segment] + fraction * (points[segment + 1] - points[segment])
            deviation = float(np.linalg.norm(data.qpos[:2] - projection))
            rows.append({
                "time_s": (step + 1) * dt,
                "x_m": float(data.qpos[0]),
                "y_m": float(data.qpos[1]),
                "error": error,
                "line_seen": error is not None,
                "steering_rad_s": float(steering),
                "deviation_m": deviation,
                "progress_m": float(progress),
            })
            # A traversal must stay near its ordered branch and finish within the time limit.
            if deviation > args.max_deviation:
                reason = "off_track"
                break
            if progress >= lengths[-1] - 0.03:
                reason = "completed"
                break
            if viewer is not None:
                if not viewer.is_running():
                    reason = "viewer_closed"
                    break
                viewer.sync()
            if args.camera:
                import cv2

                cv2.imshow(
                    "PID camera: red detection, green centroid",
                    camera_overlay(frame, error, mask)[:, :, ::-1],
                )
                if cv2.waitKey(1) & 0xFF == 27:
                    reason = "viewer_closed"
                    break
            if not args.headless:
                time.sleep(max(0, dt - (time.monotonic() - wall_start)))
    if args.camera:
        import cv2

        cv2.destroyAllWindows()
    # Preserve failures as well as successes so the baseline cannot hide poor episodes.
    with (output / f"trajectory_{seed}.csv").open("w", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "seed": seed,
        "reason": reason,
        "success": reason == "completed",
        "progress_fraction": float(progress / lengths[-1]),
        "duration_s": rows[-1]["time_s"] if rows else 0.0,
        "mean_deviation_cm": float(np.mean([r["deviation_m"] for r in rows]) * 100)
        if rows
        else None,
        "max_deviation_cm": max((r["deviation_m"] * 100 for r in rows), default=None),
        "line_loss_frames": sum(not r["line_seen"] for r in rows) + (reason == "line_lost"),
        "spawn_lateral_m": lateral,
        "spawn_yaw_deg": math.degrees(yaw_offset),
        "camera_world_m": initial_camera.tolist(),
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=[*TRACKS, "all"], default="circle")
    parser.add_argument("--list-tracks", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--controller", choices=["pid", "bang_bang"], default="pid")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--speed", type=float, default=3.0, help="forward wheel speed, rad/s")
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--camera-tilt", type=float, default=45)
    parser.add_argument("--max-deviation", type=float, default=0.06, help="base-origin error, m")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/pid") / time.strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args()
    if args.list_tracks:
        print("\n".join(TRACKS))
        return
    if args.episodes < 1 or args.duration <= 0 or not 0 < args.speed < 10:
        parser.error("episodes/duration must be positive and speed must be between 0 and 10")
    if args.max_deviation <= 0 or not 0 <= args.camera_tilt <= 90:
        parser.error("max-deviation must be positive and camera-tilt must be in [0, 90]")
    # Save one configuration and a track-by-track summary beside the raw trajectories.
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "output": str(args.output),
        "line_width_m": LINE_WIDTH,
        "camera_resolution": [CAM_RES, CAM_RES],
        "camera_fovy_deg": 60,
        "camera_position_local_m": [0, -0.108, -0.0243587 + 0.05],
        "line_loss_timeout_s": 0.5,
        "finish_tolerance_m": 0.03,
        "control_hz": CONTROL_HZ,
        "mujoco_version": mujoco.__version__,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {}
    tracks = list(TRACKS) if args.track == "all" else [args.track]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(tracks), figsize=(5 * len(tracks), 5), squeeze=False)
    for axis, track in zip(axes[0], tracks):
        waypoints = TRACKS[track]()
        model = build_model(waypoints, args.camera_tilt)
        output = args.output / track
        output.mkdir(exist_ok=True)
        outcomes = []
        points = np.asarray(waypoints)
        axis.plot(points[:, 0], points[:, 1], "k--", label="track center")
        for seed in range(args.seed, args.seed + args.episodes):
            outcome, rows = run_episode(model, waypoints, args, seed, output)
            outcomes.append(outcome)
            axis.plot([r["x_m"] for r in rows], [r["y_m"] for r in rows], alpha=0.6)
            print(
                f"{track} seed={seed} {outcome['reason']} "
                f"progress={outcome['progress_fraction']:.1%}",
                flush=True,
            )
        success = sum(o["success"] for o in outcomes) / len(outcomes)
        summary[track] = {"success_rate": success, "episodes": outcomes}
        axis.set(title=f"{track}: {success:.0%} completed", xlabel="x (m)", ylabel="y (m)")
        axis.set_aspect("equal")
        axis.legend()
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure.tight_layout()
    figure.savefig(args.output / "trajectories.png", dpi=160)
    plt.close(figure)
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
