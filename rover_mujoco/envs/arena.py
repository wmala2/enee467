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

# Rover footprint, read off rover.xml: the chassis collision proxy is 0.09 half-width, the
# wheels stick out slightly further, and the body is 0.125 long from centre to end.
ROVER_HALF_WIDTH = 0.093
ROVER_HALF_LENGTH = 0.125
# What the rover sweeps turning in place. Every clearance below is derived from this rather
# than picked by eye, so the guarantees actually hold instead of happening to hold.
ROVER_RADIUS = math.hypot(ROVER_HALF_WIDTH, ROVER_HALF_LENGTH)  # ~0.156 m

# How close the rover's centre has to get for the goal to count as reached. Lives here rather
# than in goal_nav_env.py because CLEARANCE_GOAL below is derived from it.
GOAL_TOLERANCE_M = 0.25

# Sampling margins, all in meters.
WALL_MARGIN = 0.35  # keep spawns/goals off the walls
GOAL_MIN_DISTANCE = 1.0  # a goal closer than this is not much of a task
CLEARANCE_MARGIN = 0.05  # slack on top of each derived minimum

# The spawn must be collision-free with room to turn around in.
CLEARANCE_START = ROVER_RADIUS + 0.28  # ~0.44 m

# Every pose that counts as "arrived" must also be collision-free, or the rover can satisfy
# the goal and hit something on the same step -- worth +SUCCESS_BONUS and -COLLISION_PENALTY
# at once, which is a contradictory training signal. That needs the arrival disc *plus* the
# body radius to be clear, not just the body radius: at CLEARANCE_GOAL=0.35 this was violated
# in a measured 25.2% of episodes.
CLEARANCE_GOAL = GOAL_TOLERANCE_M + ROVER_RADIUS + CLEARANCE_MARGIN  # ~0.46 m

# Two obstacles must never form a gap the rover cannot fit through, or a layout can be
# reachable on paper and impassable in practice.
CLEARANCE_BETWEEN = 2 * ROVER_HALF_WIDTH + CLEARANCE_MARGIN  # ~0.24 m

# Clearances are compared against an obstacle's *corner*, not its nominal radius. The pool
# holds boxes as well as cylinders and `radius` is the box half-extent, so a box's corner
# reaches radius*sqrt(2) -- treating every obstacle as a disc of that size is conservative for
# cylinders and correct for boxes. Without this, a big box could sit ~0.075 m closer to the
# goal than CLEARANCE_GOAL claimed (measured: 0.293 m against a stated 0.35 m).
CORNER_FACTOR = math.sqrt(2.0)

# Obstacles placed uniformly in a 5x5 m arena almost never end up in the way: measured, the
# straight line from spawn to goal was blocked in only 19.9% of episodes, so four episodes in
# five were "drive to a point" with scenery. This fraction of each layout is instead sampled
# along the direct route, which takes that to ~85% and makes avoidance the actual task.
PATH_OBSTACLE_FRACTION = 0.6
PATH_SPAN = (0.25, 0.80)  # where along the route (as a fraction of its length) they can land
PATH_LATERAL_STD = 0.22  # m: sideways scatter off the route, so it is blocked but not walled

# Lidar, mirroring the real rover's three-beam sensor (rover_control/encoder_poller.py).
LIDAR_MAX_RANGE = 2.0  # m: beyond this the env reports "nothing there", as a real sensor does

# A real time-of-flight beam is a cone (the VL53L0X-class parts used on rovers like this are
# ~25 deg), but MuJoCo's <rangefinder> casts one infinitely thin ray. Modelling each beam as a
# single ray therefore understates the hardware badly and leaves wide blind wedges between the
# beams: measured, 84% of collisions happened with no beam reading anything under 0.6 m
# beforehand, and 16% with all three still reading max range at the moment of impact. The rover
# was mostly hitting things it could not see. Each beam is now a fan of LIDAR_RAYS_PER_BEAM
# rays spanning LIDAR_FOV_DEG, reported as the minimum -- which is what a ToF part returns.
LIDAR_FOV_DEG = 25.0
LIDAR_RAYS_PER_BEAM = 5
# Splay equals the FOV so the three cones tile without a gap between them. The real sensor's
# mounting angles are an assumption to reconcile against hardware; contiguous coverage is the
# defensible default until then.
LIDAR_SPLAY_DEG = LIDAR_FOV_DEG
LIDAR_BEAM_NAMES = ("left", "center", "right")


def lidar_ray_angles():
    """Angle of every ray, in degrees left of straight ahead, grouped per beam (left, centre,
    right). Shared by scripts/gen_rover_lidar.py, which writes the sites into rover.xml, and by
    GoalNavEnv, which folds each fan back down to one distance."""
    half = LIDAR_FOV_DEG / 2.0
    step = LIDAR_FOV_DEG / (LIDAR_RAYS_PER_BEAM - 1)
    return [
        [centre - half + i * step for i in range(LIDAR_RAYS_PER_BEAM)]
        for centre in (LIDAR_SPLAY_DEG, 0.0, -LIDAR_SPLAY_DEG)
    ]


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


def sample_goal(rng, start_xy, min_distance=GOAL_MIN_DISTANCE, max_distance=None):
    """A goal position inside the arena, `min_distance` to `max_distance` metres from
    `start_xy` and kept WALL_MARGIN clear of the walls.

    Sampled as a bearing plus a distance rather than as a uniform point in the arena, because
    the curriculum needs to control *how far away* the goal is independently of where the
    rover happens to have spawned."""
    limit = ARENA_HALF - WALL_MARGIN
    if max_distance is None:
        max_distance = 2.0 * limit
    for _ in range(200):
        bearing = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(min_distance, max_distance)
        xy = start_xy + distance * np.array([math.cos(bearing), math.sin(bearing)])
        if np.all(np.abs(xy) <= limit):
            return xy
    # Cornered spawn with no sample that fits: head toward the middle instead of looping.
    direction = -start_xy / (np.linalg.norm(start_xy) or 1.0)
    return np.clip(start_xy + direction * min_distance, -limit, limit)


def sample_obstacles(rng, start_xy, goal_xy, count, path_fraction=PATH_OBSTACLE_FRACTION):
    """`count` (x, y, radius) obstacles inside the walls, none of them sitting on top of the
    spawn, the goal, or each other.

    `path_fraction` of them are sampled along the direct spawn-to-goal route rather than
    uniformly in the arena -- see PATH_OBSTACLE_FRACTION for why uniform placement made this a
    navigation task with decorations rather than an avoidance one. Rejection sampling with a
    retry budget rather than a solver: at these densities it lands in a handful of tries, and a
    short layout is better than a hung reset."""
    limit = ARENA_HALF - WALL_MARGIN
    placed = []

    route = np.asarray(goal_xy, dtype=float) - np.asarray(start_xy, dtype=float)
    route_len = float(np.linalg.norm(route))
    unit = route / route_len if route_len else np.array([1.0, 0.0])
    normal = np.array([-unit[1], unit[0]])
    n_on_path = round(count * path_fraction) if route_len else 0

    def acceptable(xy, radius):
        reach = radius * CORNER_FACTOR
        if np.any(np.abs(xy) > limit):
            return False
        if np.linalg.norm(xy - start_xy) < CLEARANCE_START + reach:
            return False
        if np.linalg.norm(xy - goal_xy) < CLEARANCE_GOAL + reach:
            return False
        return not any(
            np.linalg.norm(xy - other_xy) < reach + other_r * CORNER_FACTOR + CLEARANCE_BETWEEN
            for other_xy, other_r in placed
        )

    for attempt in range(count * 40):
        if len(placed) == count:
            break
        radius = rng.uniform(*OBSTACLE_RADIUS_RANGE)
        # Try the route first; fall back to uniform once the quota is met, and also once the
        # retry budget is half gone, so a cramped route can never starve the whole layout.
        on_path = len(placed) < n_on_path and attempt < count * 20
        if on_path:
            along = rng.uniform(*PATH_SPAN) * route_len
            offset = rng.normal(0.0, PATH_LATERAL_STD)
            xy = start_xy + along * unit + offset * normal
        else:
            xy = rng.uniform(-limit, limit, size=2)
        if acceptable(xy, radius):
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
