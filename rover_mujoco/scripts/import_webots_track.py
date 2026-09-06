"""Converts a Webots track PROTO into a MuJoCo MJCF fragment.

The tracks in ../webots_ws/protos are each a thin wrapper around a flat OBJ mesh with a black
material, which MuJoCo loads directly -- so the conversion is mostly bookkeeping: read the
scale the PROTO applies, work out which axis the mesh is flat in, rotate that onto Z, centre
it, and emit a visual-only geom.

    uv run python scripts/import_webots_track.py ../../webots_ws/protos/CircleTrack
    uv run python scripts/import_webots_track.py --all ../../webots_ws/protos

Output goes to assets/objects/tracks/<name>.xml alongside a copy of the mesh. Scenes include
it the same way they include the generated tracks (see assets/robots/rover/rover_line_*.xml).
"""

import argparse
import os
import re
import shutil

import numpy as np

# Lifted just clear of the floor plane. A mesh at exactly z=0 z-fights with the floor and
# renders as a flickering mess, which the camera then sees as noise rather than a line.
TRACK_Z = 0.002
# Group 3, matching the generated tracks: LineFollowerEnv finds the line geoms by this group
# to randomize their shade, and the viewers enable it explicitly.
TRACK_GROUP = 3


def parse_proto(proto_dir):
    """(obj path, scale) from the PROTO in `proto_dir`."""
    protos = [f for f in os.listdir(proto_dir) if f.endswith(".proto")]
    if not protos:
        raise SystemExit(f"no .proto in {proto_dir}")
    with open(os.path.join(proto_dir, protos[0]), encoding="utf-8", errors="ignore") as f:
        text = f.read()
    match = re.search(r'"([^"]+\.obj)"', text)
    if not match:
        raise SystemExit(f"no .obj referenced by {protos[0]}")
    scale_match = re.search(r"scale\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    scale = np.array([float(v) for v in scale_match.groups()]) if scale_match else np.ones(3)
    return os.path.join(proto_dir, match.group(1)), scale


def mesh_bounds(obj_path):
    with open(obj_path, encoding="utf-8", errors="ignore") as f:
        verts = [[float(v) for v in line.split()[1:4]] for line in f if line.startswith("v ")]
    if not verts:
        raise SystemExit(f"no vertices in {obj_path}")
    verts = np.array(verts)
    return verts.min(axis=0), verts.max(axis=0)


def convert(proto_dir, out_dir):
    name = os.path.basename(os.path.normpath(proto_dir))
    obj_path, scale = parse_proto(proto_dir)
    lo, hi = mesh_bounds(obj_path)
    extent = (hi - lo) * scale
    centre = (hi + lo) / 2 * scale

    # Which axis is the sheet's normal? Webots tracks are flat, but not all in the same plane.
    flat = int(np.argmin(extent))
    if flat == 2:
        quat = "1 0 0 0"  # already lying in XY
    elif flat == 1:
        quat = "0.70710678 0.70710678 0 0"  # rotate Y onto Z
    else:
        quat = "0.70710678 0 0.70710678 0"  # rotate X onto Z

    os.makedirs(out_dir, exist_ok=True)
    # MuJoCo resolves <mesh file=...> against the *main* XML's directory, not the file doing
    # the including -- same as rover.xml's "meshes/left_motor.stl". Every scene that includes a
    # track lives in assets/robots/rover/, so the path is written relative to there.
    mesh_file = f"../../objects/tracks/{name}.obj"
    shutil.copy(obj_path, os.path.join(out_dir, f"{name}.obj"))

    planar = sorted(extent)[-2:]
    xml = f"""<mujoco model="{name}">
  <!-- Converted from {os.path.basename(proto_dir)}'s Webots PROTO by
       scripts/import_webots_track.py. Real-world size {planar[1]:.2f} x {planar[0]:.2f} m
       (the PROTO applies scale {scale[0]:g} {scale[1]:g} {scale[2]:g} to the raw mesh).
       Visual only -- contype/conaffinity 0 -- because a track is paint on the floor, not
       something to drive into. Group {TRACK_GROUP} so LineFollowerEnv's appearance
       randomization finds it and the viewers show it. -->
  <asset>
    <!-- inertia="shell": these meshes are perfectly flat, so MuJoCo's default volume-based
         inertia computation rejects them ("mesh volume is too small"). Shell inertia treats
         the mesh as a surface, which is what a painted track is. It never actually matters
         here -- the geom is static and visual-only -- but the compiler still validates it. -->
    <mesh name="{name}" file="{mesh_file}" inertia="shell"
          scale="{scale[0]:g} {scale[1]:g} {scale[2]:g}"/>
  </asset>
  <worldbody>
    <geom name="{name}" type="mesh" mesh="{name}" quat="{quat}"
          pos="{-centre[0]:.4f} {-centre[1]:.4f} {TRACK_Z}"
          contype="0" conaffinity="0" group="{TRACK_GROUP}" rgba="0.05 0.05 0.05 1"/>
  </worldbody>
</mujoco>
"""
    out_path = os.path.join(out_dir, f"{name}.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"  {name:18s} {planar[1]:5.2f} x {planar[0]:5.2f} m -> {os.path.basename(out_path)}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="a PROTO directory, or the protos root with --all")
    p.add_argument("--all", action="store_true", help="convert every track under `path`")
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "../assets/objects/tracks"),
        help="where to write the MJCF and mesh",
    )
    args = p.parse_args()

    dirs = (
        [
            os.path.join(args.path, d)
            for d in sorted(os.listdir(args.path))
            if os.path.isdir(os.path.join(args.path, d))
            and any(f.endswith(".proto") for f in os.listdir(os.path.join(args.path, d)))
        ]
        if args.all
        else [args.path]
    )
    print(f"converting {len(dirs)} track(s):")
    for d in dirs:
        try:
            convert(d, args.out)
        except SystemExit as exc:
            print(f"  {os.path.basename(d):18s} skipped: {exc}")


if __name__ == "__main__":
    main()
