"""Generates a black line-track MJCF fragment (a polyline of thin box geoms lying flat
on the floor) from a list of (x, y) waypoints. Re-run this to regenerate a track after
tweaking its shape below; the output is a worldbody-only <mujoco> file meant to be
pulled in via <include> from a scene (see assets/robots/rover/rover_line_*.xml).
"""

import math
import os

from envs.tracks import oval_waypoints
from envs.tracks import s_curve_waypoints

# Visual/collision-free line appearance
LINE_WIDTH = 0.03
LINE_THICKNESS = 0.001
LINE_RGBA = "0.05 0.05 0.05 1"


def build_track_xml(name, waypoints):
    lines = [f'<mujoco model="{name}">', "  <worldbody>"]
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        seg_len = math.hypot(x1 - x0, y1 - y0)
        heading_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
        lines.append(
            f'    <geom type="box" size="{seg_len / 2:.4f} {LINE_WIDTH / 2} {LINE_THICKNESS}" '
            f'pos="{mx:.4f} {my:.4f} {LINE_THICKNESS}" euler="0 0 {heading_deg:.2f}" '
            f'contype="0" conaffinity="0" group="3" rgba="{LINE_RGBA}"/>'
        )
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "../assets/objects/tracks")
    os.makedirs(out_dir, exist_ok=True)

    tracks = {
        "track_oval": oval_waypoints(),
        "track_s_curve": s_curve_waypoints(),
    }
    for name, waypoints in tracks.items():
        path = os.path.join(out_dir, f"{name}.xml")
        with open(path, "w") as f:
            f.write(build_track_xml(name, waypoints))
        print(f"wrote {path} ({len(waypoints)} waypoints)")
