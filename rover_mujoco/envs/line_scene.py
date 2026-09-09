"""Shared CAD rover, camera, and tape geometry for classical and learned controllers."""

import itertools
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco

# Group 2 is deliberate: MjvOption's default mask is [1, 1, 1, 0, 0, 0], so anything in
# group 3+ is invisible to every renderer unless the call site re-enables it, silently.
LINE_GROUP = 2
GENERATED_LINE_WIDTH = 0.03
LINE_THICKNESS = 0.001
LINE_RGBA = "0.05 0.05 0.05 1"


def build_track_xml(name, waypoints, line_width=GENERATED_LINE_WIDTH):
    lines = [f'<mujoco model="{name}">', "  <worldbody>"]
    for i, ((x0, y0), (x1, y1)) in enumerate(itertools.pairwise(waypoints)):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        seg_len = math.hypot(x1 - x0, y1 - y0)
        heading = math.atan2(y1 - y0, x1 - x0)
        # Quaternion, not euler: rover.xml sets <compiler angle="radian">, so euler degrees
        # are read as radians and the track comes out as scattered dashes.
        qw, qz = math.cos(heading / 2), math.sin(heading / 2)
        # Half-length is padded by half the line width so consecutive boxes overlap instead of
        # merely touching; without it every bend leaves a visible wedge on its outside.
        half_len = seg_len / 2 + line_width / 2
        lines.append(
            f'    <geom name="line_{i}" type="box" '
            f'size="{half_len:.4f} {line_width / 2} {LINE_THICKNESS}" '
            f'pos="{mx:.4f} {my:.4f} {LINE_THICKNESS}" quat="{qw:.6f} 0 0 {qz:.6f}" '
            f'contype="0" conaffinity="0" group="{LINE_GROUP}" rgba="{LINE_RGBA}"/>'
        )
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


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
