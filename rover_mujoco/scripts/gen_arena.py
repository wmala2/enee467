"""Generates the walled 5x5 m arena and its pool of movable obstacles as an MJCF fragment
(assets/objects/arena/arena_5x5.xml). Re-run this after changing the constants in
envs/arena.py; the output is a worldbody-only <mujoco> file meant to be pulled in via
<include> from a scene (see assets/robots/rover/rover_arena_real.xml).

The obstacles are emitted at their parking spot outside the walls with a placeholder size:
GoalNavEnv positions and resizes them every reset, since MuJoCo compiles geometry once but
the whole point of the task is a different layout each episode.

Usage: uv run python scripts/gen_arena.py
"""

import os

from envs import arena

# Obstacle geoms carry this group so the env can find them by group rather than by name,
# the same way LineFollowerEnv picks out the line geoms.
OBSTACLE_GROUP = 2  # visible by default; see gen_track.py's LINE_GROUP

# Bit 1 keeps these colliding with the wheels (default contype/conaffinity 1); bit 2 is what
# rover.xml's chassis_collision proxy uses, so the rover's body can't pass through them.
COLLIDE_BITS = 'contype="3" conaffinity="3"'


def build_arena_xml():
    lines = ['<mujoco model="arena_5x5">', "  <worldbody>"]
    lines.append(  # noqa: FURB113 -- one long comment block, not a run of appends
        f"    <!-- {arena.ARENA_SIZE:g}x{arena.ARENA_SIZE:g} m arena. Deliberately unwalled: the\n"
        "         boundary is a soft one that the rover can cross, which ends the episode as a\n"
        "         failure (OUT_OF_BOUNDS_TOLERANCE). Walls would make straying impossible and\n"
        "         would also hand the camera a free, always-present visual reference. -->"
    )

    lines.append(
        "    <!-- Obstacle pool: parked outside the walls, moved into place each reset\n"
        "         (see envs/tasks/goal_nav_env.py). Each is a MOCAP body rather than a bare\n"
        "         worldbody geom, and that is load-bearing: MuJoCo precomputes broadphase\n"
        "         bounding volumes for static (world) geoms, so one moved at runtime via\n"
        "         model.geom_pos still renders and ray-casts at its new position but collides\n"
        "         at its old one -- the rover drove clean through every obstacle. Mocap bodies\n"
        "         are program-movable and go through the dynamic broadphase, so contacts\n"
        "         follow them. -->"
    )
    park_x, park_y = arena.OBSTACLE_PARKING_XY
    for i in range(arena.MAX_OBSTACLES):
        # Alternating shapes so a policy sees both flat faces and curved ones.
        is_box = i % 2 == 0
        radius = sum(arena.OBSTACLE_RADIUS_RANGE) / 2
        half_h = arena.OBSTACLE_HEIGHT / 2
        if is_box:
            kind, size = "box", f"{radius} {radius} {half_h}"
        else:
            kind, size = "cylinder", f"{radius} {half_h}"
        body_open = (
            f'    <body name="obstacle_{i}" mocap="true"'
            f' pos="{park_x + i * 0.5} {park_y} {half_h}">'
        )
        geom = (
            f'      <geom name="obstacle_{i}" type="{kind}" group="{OBSTACLE_GROUP}"'
            f' size="{size}" {COLLIDE_BITS} rgba="0.35 0.35 0.4 1"/>'
        )
        lines.extend([body_open, geom, "    </body>"])

    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "../assets/objects/arena")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "arena_5x5.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_arena_xml())
    print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
