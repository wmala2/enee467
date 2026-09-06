"""Geometry for the goal-navigation task: the 5x5 m arena, where obstacles and goals get
sampled inside it, and the dead-reckoning odometry the policy navigates by.

Shared by scripts/gen_arena.py (which turns the constants below into MJCF) and
envs/tasks/goal_nav_env.py (which needs the matching numbers at runtime), the same way
envs/tracks.py is shared by gen_track.py and the line-follower envs.
"""

import math

import numpy as np

# Arena: a 5x5 m square with walls, matching the "work inside a 5x5 meter space" spec.
ARENA_SIZE = 5.0
ARENA_HALF = ARENA_SIZE / 2.0
WALL_HEIGHT = 0.15
WALL_THICKNESS = 0.05

# Obstacle pool. MuJoCo compiles geometry once, so the scene declares a fixed pool and the
# env moves/resizes/hides them each reset (model.geom_pos/geom_size, the same runtime-mutation
# trick LineFollowerEnv's domain randomization uses on geom_rgba). Half boxes, half cylinders
# so the policy sees both flat faces and curved ones.
MAX_OBSTACLES = 8
OBSTACLE_HEIGHT = 0.12
OBSTACLE_RADIUS_RANGE = (0.05, 0.18)
# Where an unused obstacle goes: outside the walls, so it is neither visible nor collidable.
OBSTACLE_PARKING_XY = (ARENA_HALF + 5.0, ARENA_HALF + 5.0)

# Sampling margins, all in meters.
WALL_MARGIN = 0.35  # keep spawns/goals off the walls
GOAL_MIN_DISTANCE = 1.0  # a goal closer than this is not much of a task
CLEARANCE_START = 0.45  # obstacle-free radius around the rover's spawn
CLEARANCE_GOAL = 0.35  # ...and around the goal, so it is always reachable
CLEARANCE_BETWEEN = 0.30  # minimum gap between two obstacles

# Lidar, mirroring the real rover's three-beam sensor (rover_control/encoder_poller.py).
LIDAR_SPLAY_DEG = 30.0  # outer beams either side of straight ahead; matches rover.xml's sites
LIDAR_MAX_RANGE = 2.0  # m: beyond this the env reports "nothing there", as a real sensor does
LIDAR_NAMES = ("lidar_left", "lidar_center", "lidar_right")

# Differential-drive geometry, read off rover.xml's wheel body positions
# (left_wheel x=+0.0798, right_wheel x=-0.0828) and the wheel mesh radius.
WHEEL_BASE = 0.1626
WHEEL_RADIUS = 0.0335


def sample_start(rng):
    """A spawn (x, y) and yaw anywhere in the arena, kept WALL_MARGIN clear of the walls.
    Spawning somewhere different every episode (rather than always at the origin facing +Y)
    is what stops the policy from memorizing one absolute layout."""
    limit = ARENA_HALF - WALL_MARGIN
    return rng.uniform(-limit, limit, size=2), rng.uniform(-math.pi, math.pi)


def sample_goal(rng, start_xy):
    """A goal position inside the arena at least GOAL_MIN_DISTANCE from `start_xy`, kept
    WALL_MARGIN clear of the walls."""
    limit = ARENA_HALF - WALL_MARGIN
    for _ in range(200):
        xy = rng.uniform(-limit, limit, size=2)
        if np.linalg.norm(xy - start_xy) >= GOAL_MIN_DISTANCE:
            return xy
    # Cornered spawn with no far-enough sample: step out toward the middle instead of looping.
    direction = -start_xy / (np.linalg.norm(start_xy) or 1.0)
    return np.clip(start_xy + direction * GOAL_MIN_DISTANCE, -limit, limit)


def sample_obstacles(rng, start_xy, goal_xy, count):
    """`count` (x, y, radius) obstacles inside the walls, none of them sitting on top of the
    spawn, the goal, or each other. Rejection sampling with a retry budget rather than a
    solver: at these densities it lands in a handful of tries, and a short layout is better
    than a hung reset."""
    limit = ARENA_HALF - WALL_MARGIN
    placed = []
    for _ in range(count * 30):
        if len(placed) == count:
            break
        radius = rng.uniform(*OBSTACLE_RADIUS_RANGE)
        xy = rng.uniform(-limit, limit, size=2)
        if np.linalg.norm(xy - start_xy) < CLEARANCE_START + radius:
            continue
        if np.linalg.norm(xy - goal_xy) < CLEARANCE_GOAL + radius:
            continue
        if any(
            np.linalg.norm(xy - other_xy) < radius + other_r + CLEARANCE_BETWEEN
            for other_xy, other_r in placed
        ):
            continue
        placed.append((xy, radius))
    return placed


def integrate_odometry(pose, left_ticks, right_ticks, counts_per_rev):
    """Dead-reckon one control step of differential drive from wheel encoder ticks.

    `pose` is the current (x, y, yaw) estimate and the return value is the next one. This is
    deliberately the same arc-free approximation a student would write on the real rover --
    treat the step as "turn by dtheta, then translate by ds" -- because the point of this
    estimate is that it drifts: it is fed the same noisy, quantized ticks the policy sees, so
    the goal vector the policy navigates by degrades exactly like the real one would.
    """
    x, y, yaw = pose
    left_rad = 2.0 * math.pi * left_ticks / counts_per_rev
    right_rad = 2.0 * math.pi * right_ticks / counts_per_rev
    # Two sign conventions from rover.xml have to be undone here, and getting either wrong
    # flips the turn direction (measured: a driving arc came out mirrored about the rover's
    # axis until both were fixed):
    #   1. The right wheel's body frame is mirrored, so equal-sign wheel commands spin in
    #      place; negating it recovers the usual differential-drive mixing.
    #   2. The joint *names* are swapped relative to the physical sides -- "left_axle" is the
    #      wheel at x=+0.0798, which is on the rover's right when forward is +Y. So the ticks
    #      that arrive as `left_ticks` belong to the physically-right wheel.
    right_wheel_dist = left_rad * WHEEL_RADIUS
    left_wheel_dist = -right_rad * WHEEL_RADIUS
    ds = (left_wheel_dist + right_wheel_dist) / 2.0
    dyaw = (right_wheel_dist - left_wheel_dist) / WHEEL_BASE
    # Midpoint integration -- advance half the turn, translate along that heading, then the
    # other half. The naive "turn fully, then translate" version is a line shorter but bakes
    # in a systematic bias on arcs (measured 0.16 m of forward error over a six-second turn,
    # with no noise at all), and the drift here is supposed to come from the sensor, not from
    # the integrator being sloppy.
    yaw += dyaw / 2.0
    x += ds * math.cos(yaw)
    y += ds * math.sin(yaw)
    yaw += dyaw / 2.0
    return np.array([x, y, yaw])


def goal_in_body_frame(pose, goal_xy):
    """The goal as the policy sees it: (forward, left) meters in the rover's *estimated* body
    frame. This is what makes "drive to (1, 2)" a well-posed command with no privileged
    state -- the rover knows where it thinks it is, not where it is."""
    x, y, yaw = pose
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return np.array([forward, left], dtype=np.float32)
