"""Writes the lidar ray sites into rover.xml and the matching rangefinder sensors into the
goal-navigation scene, from the beam geometry in envs/arena.py.

Each of the rover's three time-of-flight beams is a cone, not a ray, but MuJoCo's <rangefinder>
casts a single infinitely thin ray -- so a beam is modelled as a fan of rays and folded back to
one distance by taking the minimum, which is what a real ToF part reports. That means 3 x
LIDAR_RAYS_PER_BEAM sites and sensors, which is more than anyone wants to keep in sync by hand.

Usage: uv run python scripts/gen_rover_lidar.py
"""

import math
import os

from envs import arena

ROVER_XML = "../assets/robots/rover/rover.xml"
SCENE_XML = "../assets/robots/rover/rover_arena_real.xml"

SITE_BEGIN = "      <!-- BEGIN generated lidar sites (scripts/gen_rover_lidar.py) -->"
SITE_END = "      <!-- END generated lidar sites -->"
SENSOR_BEGIN = "    <!-- BEGIN generated lidar sensors (scripts/gen_rover_lidar.py) -->"
SENSOR_END = "    <!-- END generated lidar sensors -->"

# Where the sensor sits on the chassis: forward of centre and low, roughly where the real part
# is mounted. Robot forward is local +Y (see teleop_rover.py).
MOUNT_POS = "0 0.13 -0.06"


def site_lines():
    fov_note = (
        f"           LIDAR_FOV_DEG ({arena.LIDAR_FOV_DEG:g} deg); GoalNavEnv takes the minimum"
        " over each fan,"
    )
    lines = [
        SITE_BEGIN,
        "      <!-- The rover's three forward ToF beams. Each is a fan of rays spanning",
        fov_note,
        "           which is what a real cone-FOV part reports. A rangefinder casts along its",
        "           site's local +Z, so zaxis aims each ray. Massless, collision-free sites:",
        "           scenes without matching <rangefinder> sensors pay nothing for them. -->",
    ]
    for name, angles in zip(arena.LIDAR_BEAM_NAMES, arena.lidar_ray_angles(), strict=True):
        for i, deg in enumerate(angles):
            rad = math.radians(deg)
            # Positive angle = toward the rover's left, which is -X when forward is +Y.
            lines.append(
                f'      <site name="lidar_{name}_{i}" pos="{MOUNT_POS}" size="0.005"'
                f' zaxis="{-math.sin(rad):.6f} {math.cos(rad):.6f} 0"/>'
            )
    lines.append(SITE_END)
    return lines


def sensor_lines():
    lines = [
        SENSOR_BEGIN,
        "    <!-- One rangefinder per lidar ray. Ordered left/centre/right to match the",
        '         firmware\'s "l" reply, so the observation lines up field-for-field with what',
        "         rover_control/encoder_poller.py's lidar_latest() returns on the real rover. -->",
    ]
    for name, angles in zip(arena.LIDAR_BEAM_NAMES, arena.lidar_ray_angles(), strict=True):
        for i in range(len(angles)):
            lines.append(f'    <rangefinder name="range_{name}_{i}" site="lidar_{name}_{i}"/>')
    lines.append(SENSOR_END)
    return lines


def splice(path, begin, end, new_lines):
    """Replace everything between the markers, inserting them if not present yet."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if begin in lines:
        head = lines[: lines.index(begin)]
        tail = lines[lines.index(end) + 1 :]
    else:
        raise SystemExit(f"markers not found in {path}; add {begin!r} / {end!r} first")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + new_lines + tail))


def main():
    here = os.path.dirname(__file__)
    splice(os.path.join(here, ROVER_XML), SITE_BEGIN, SITE_END, site_lines())
    splice(os.path.join(here, SCENE_XML), SENSOR_BEGIN, SENSOR_END, sensor_lines())
    n = 3 * arena.LIDAR_RAYS_PER_BEAM
    print(f"wrote {n} lidar sites to rover.xml and {n} rangefinder sensors to the arena scene")


if __name__ == "__main__":
    main()
