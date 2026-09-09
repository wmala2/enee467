"""Generates a black line-track MJCF fragment (a polyline of thin box geoms lying flat
on the floor) from a list of (x, y) waypoints. Re-run this to regenerate a track after
tweaking its shape below; the output is a worldbody-only <mujoco> file meant to be
pulled in via <include> from a scene (see assets/robots/rover/rover_line_*.xml).
"""

import itertools
import math
import os

from envs.tracks import circle_waypoints
from envs.tracks import hairpin_waypoints
from envs.tracks import oval_waypoints
from envs.tracks import s_curve_waypoints
from envs.tracks import wave_waypoints

# Visual/collision-free line appearance. Group 2 is deliberate: MjvOption's default
# geomgroup mask is [1, 1, 1, 0, 0, 0], so anything in group 3+ is invisible to every
# renderer unless each call site remembers to switch it back on. The track used to be
# group 3, and the omission was silent -- the onboard camera rendered bare floor and the
# dark-pixel test locked onto the sky instead, which invalidated a "finds the line 12/12"
# check and a "100% completion" policy that was really driving blind.
LINE_GROUP = 2
LINE_WIDTH = 0.03
LINE_THICKNESS = 0.001
LINE_RGBA = "0.05 0.05 0.05 1"


def build_track_xml(name, waypoints):
    lines = [f'<mujoco model="{name}">', "  <worldbody>"]
    for i, ((x0, y0), (x1, y1)) in enumerate(itertools.pairwise(waypoints)):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        seg_len = math.hypot(x1 - x0, y1 - y0)
        heading = math.atan2(y1 - y0, x1 - x0)
        # Quaternion rather than euler, deliberately. This used to emit `euler="0 0 <degrees>"`
        # while rover.xml sets a global <compiler angle="radian">, so MuJoCo read each heading
        # as radians -- a 45-degree segment was rotated 45 radians. Every box landed at an
        # essentially arbitrary angle and the "line" came out as scattered dashes. A quaternion
        # carries no unit, so the footgun cannot recur.
        qw, qz = math.cos(heading / 2), math.sin(heading / 2)
        # Half-length is padded by half the line width so consecutive boxes overlap instead of
        # merely touching; without it every bend leaves a visible wedge on its outside.
        half_len = seg_len / 2 + LINE_WIDTH / 2
        lines.append(
            f'    <geom name="line_{i}" type="box" '
            f'size="{half_len:.4f} {LINE_WIDTH / 2} {LINE_THICKNESS}" '
            f'pos="{mx:.4f} {my:.4f} {LINE_THICKNESS}" quat="{qw:.6f} 0 0 {qz:.6f}" '
            f'contype="0" conaffinity="0" group="{LINE_GROUP}" rgba="{LINE_RGBA}"/>'
        )
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "../assets/objects/tracks")
    os.makedirs(out_dir, exist_ok=True)

    tracks = {
        # Training tracks
        "track_oval": oval_waypoints(),
        "track_s_curve": s_curve_waypoints(),
        # Held-out evaluation tracks (see envs/tracks.py) -- generated the same way, but the
        # envs expose them separately so training can never sample them.
        "track_circle": circle_waypoints(),
        "track_wave": wave_waypoints(),
        "track_hairpin": hairpin_waypoints(),
    }
    for name, waypoints in tracks.items():
        path = os.path.join(out_dir, f"{name}.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_track_xml(name, waypoints))
        print(f"wrote {path} ({len(waypoints)} waypoints)")
