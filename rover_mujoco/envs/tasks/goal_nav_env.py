"""Goal navigation with obstacle avoidance, on the same sim-to-real footing as
LineFollowerRealEnv: drive to a commanded (x, y) offset inside a 5x5 m arena without hitting
anything, using only what the physical rover can actually sense.

The command is a coordinate, not something visible -- "drive to (1, 2)" means 1 m to the
rover's right and 2 m forward of wherever it started. Nothing in the world marks that spot,
so the only way to find it is to dead-reckon from the wheel encoders, which is exactly the
point: the goal vector the policy is handed drifts, because it is integrated from the same
noisy quantized ticks the policy sees. It is a *pose* goal -- "heading North" fixes the final
orientation too -- and obstacle avoidance is camera-only, so the rover can see obstacles but
never their range. See docs/rl-goal-nav.md.
"""

from collections import deque
import math
import os
from typing import ClassVar

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from envs import arena
from envs import motor
from envs.camera import set_camera_tilt
from envs.tasks.line_follower_env import CAM_RES
from envs.tasks.line_follower_env import CONTROL_HZ
from envs.tasks.line_follower_env import FALL_HEIGHT
from envs.tasks.line_follower_real_env import _pixelate
from envs.tasks.line_follower_real_env import ACTION_LATENCY_STEPS_RANGE
from envs.tasks.line_follower_real_env import ACTION_NOISE_STD_RANGE
from envs.tasks.line_follower_real_env import BRIGHTNESS_RANGE
from envs.tasks.line_follower_real_env import ENCODER_CPR_WHEEL
from envs.tasks.line_follower_real_env import ENCODER_NOISE_STD_RANGE
from envs.tasks.line_follower_real_env import ENCODER_TICKS_BOUND
from envs.tasks.line_follower_real_env import FOVY_RANGE
from envs.tasks.line_follower_real_env import PIXEL_NOISE_STD_RANGE
from envs.tasks.line_follower_real_env import RESOLUTION_LEVELS
from envs.tasks.line_follower_real_env import WHEEL_FRICTION_RANGE
from envs.tasks.line_follower_real_env import WHITE_BALANCE_RANGE

SCENE_FILE = "rover_arena_real.xml"

# Forward-facing, matching the physical mount. Measured fraction of the 64x64 frame that an
# obstacle changes, by tilt and distance:
#     tilt   0.3 m   0.5 m   0.8 m   1.2 m
#     15.2    0.0%    0.0%    0.0%    0.0%   (the line follower's near-straight-down mount)
#     60     47.9%   16.1%    3.0%    1.6%
#     90     25.6%   16.0%    2.7%    1.2%
# Avoidance from this camera is therefore inherently short-range and reactive: usable warning
# inside ~0.5 m (about 1.5 s at top speed), effectively none past 0.8 m at any tilt.
#
# 60 rather than a fully-forward 90 deg, which is the counter-intuitive part. The obstacles are
# 0.12 m tall and the camera sits ~0.04 m up, so pointing straight ahead frames them against
# the horizon -- the same colour as the floor plane behind them -- while tilting down-forward
# frames them against the floor. Measured separation between "empty arena" and "obstacle 0.4 m
# dead ahead", as the fraction of the frame's lower band that differs from its median:
#     tilt   empty   obstacle
#      60    0.000     0.323   <- trivially separable
#      90    0.149     0.153   <- no threshold separates these
# The information is still present at 90 deg (an obstacle changes ~26% of pixels against a
# fixed empty reference, so a CNN could learn it), but nothing classical can segment it, and
# the learning problem is much harder for no benefit.
CAMERA_ANGLE_DEG = 60.0

MAX_EPISODE_STEPS = 400  # control steps @ CONTROL_HZ = 40 simulated seconds
# Defined in envs/arena.py because the obstacle clearances are derived from it.
GOAL_TOLERANCE_M = arena.GOAL_TOLERANCE_M

# Observation scaling: the vector fields span ticks in the hundreds and metres in single
# digits, and SB3's MultiInputPolicy concatenates them raw into one MLP. Fixed constants rather
# than a running VecNormalize, so deployment applies three numbers and ships no statistics file.
ENCODER_SCALE = ENCODER_TICKS_BOUND
LIDAR_SCALE = arena.LIDAR_MAX_RANGE
GOAL_SCALE = arena.ARENA_SIZE * 1.5

# --- Reward ------------------------------------------------------------------------------
# Arriving is the largest term by design; progress is shaping that gets the policy there, not
# the objective itself. The arrival bonus is ARRIVAL_BASE (guaranteed the moment you make it,
# so a scruffier arrival is never *punished*) plus up to ARRIVAL_ACCURACY split evenly between
# how centred and how well oriented the final pose is -- so 5 mm beats 5 cm without 5 cm being
# a failure.
ARRIVAL_BASE = 100.0
ARRIVAL_ACCURACY = 100.0
PROGRESS_WEIGHT = 50.0  # summed over a whole run; secondary to arriving
COLLISION_PENALTY = 75.0  # costly, but never worth more than arriving
OUT_OF_BOUNDS_PENALTY = 50.0
STEP_PENALTY = 0.05  # per step, so faster is better and standing still is never optimal

# Reward-only proximity shaping, from the *true* distance to the nearest obstacle. Measured on
# this task: a terminal-only collision penalty stalls training at ~40% arrival for millions of
# steps because nothing warns the policy on the way in. With the lidar off by default this term
# has no matching observation -- the policy has to learn the association from camera appearance,
# which is precisely the skill being trained.
PROXIMITY_THRESHOLD_M = 0.5
PROXIMITY_WEIGHT = 1.0

# Abandon an episode that is going nowhere: net displacement under STUCK_DISTANCE_M across
# STUCK_WINDOW control steps means wedged against something or spinning on the spot.
STUCK_WINDOW = 40
STUCK_DISTANCE_M = 0.05

# Sensor history. The policy is a memoryless MLP, so a single frame cannot tell it whether an
# obstacle is closing or how fast the wheels are actually turning. Stacked inside the env
# rather than by a VecFrameStack wrapper, so evaluation and the real-rover bridge stay honest
# by keeping the same deque.
OBS_HISTORY = 4

CURRICULUM_WINDOW = 50  # episodes of history each env keeps
CURRICULUM_PROMOTE_RATE = 0.7  # arrive this often over that window and the level goes up

# Heading enters through the curriculum rather than all at once: learn to arrive, then learn to
# arrive facing the right way. Level 0's 180 deg tolerance ignores heading entirely.
CURRICULUM = (
    {"obstacles": (0, 0), "goal_distance": (0.5, 1.2), "heading_tol_deg": 180.0},
    {"obstacles": (1, 3), "goal_distance": (0.75, 1.8), "heading_tol_deg": 60.0},
    {"obstacles": (2, 5), "goal_distance": (1.0, 2.5), "heading_tol_deg": 40.0},
    {
        "obstacles": (3, arena.MAX_OBSTACLES),
        "goal_distance": (1.0, 3.5),
        "heading_tol_deg": arena.HEADING_TOLERANCE_DEG,
    },
)

# Lidar realism, used only when include_lidar=True. The real sensor answers in integer
# millimetres; MuJoCo's rangefinder returns -1 for "nothing", folded into max range here.
LIDAR_NOISE_STD_RANGE = (0.0, 0.02)
LIDAR_QUANTUM_M = 0.001
# encoder_poller.py alternates its "e" and "l" queries, so each sensor refreshes at half the
# poll rate; rl_rover.py polls encoders at the full rate, leaving the lidar every other step.
LIDAR_DECIMATION = 2


def wrap_angle(theta):
    """Fold an angle into (-pi, pi] so heading errors never jump by 2*pi."""
    return math.atan2(math.sin(theta), math.cos(theta))


class GoalNavEnv(gym.Env):
    """Drive to a commanded *pose* in a 5x5 m arena without hitting anything.

    "Go to (1, 2) heading North" means 1 m to the rover's right, 2 m forward, and ending
    turned to the heading it started at -- all in the rover's own start frame, because
    encoders are all it has and they cannot know a compass direction.

    Observation, all of it available on the real rover:
      image     -- 64x64 grayscale from the forward-facing onboard camera, through the same
                   camera-realism randomization as LineFollowerRealEnv. The only obstacle
                   sensor: monocular, so it carries no range.
      encoders  -- the last OBS_HISTORY left/right wheel tick readings, most recent first.
      goal      -- (forward, left, cos dtheta, sin dtheta) to the goal pose in the rover's
                   *estimated* start frame, dead-reckoned from those same noisy ticks. Never
                   ground truth: it drifts, and recovering from that drift is part of the task.
      lidar     -- optional (include_lidar=True), the rover's real three-beam ToF sensor.

    Action: left/right wheel velocity targets in rad/s, bounded by the real motors' envelope
    and passed through envs/motor.py's JGA25-371 model.

    Reward uses ground-truth pose. That is a training-only privilege -- reward does not exist
    at deployment, unlike the observation, which stays honest.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        domain_randomize=True,
        include_image=True,
        include_lidar=False,
        curriculum=False,
    ):
        super().__init__()
        self.domain_randomize = domain_randomize
        self.render_mode = render_mode
        self.include_image = include_image
        # The lidar is real hardware (rover_control/encoder_poller.py) and the sim models it,
        # but obstacle avoidance here is specified as camera-only. Kept behind a flag rather
        # than deleted, so the two can be compared rather than argued about.
        self.include_lidar = include_lidar
        self.curriculum = curriculum
        self._level = 0
        self._recent_arrivals = deque(maxlen=CURRICULUM_WINDOW)
        self._sensor_history = deque(maxlen=OBS_HISTORY)
        self._recent_positions = deque(maxlen=STUCK_WINDOW)

        model_path = os.path.join(
            os.path.dirname(__file__), "../../assets/robots/rover", SCENE_FILE
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = (
            mujoco.Renderer(self.model, height=CAM_RES, width=CAM_RES) if include_image else None
        )

        # Normalized to [-1, 1] and scaled to rad/s inside step(). This is not cosmetic: SB3's
        # Gaussian policy starts at std ~= 1 around zero, so an action space in physical units
        # (+/-MAX_WHEEL_SPEED = 7.46 rad/s) puts 97.5% of sampled commands inside the motors'
        # 2.24 rad/s dead zone. The rover then cannot move at all, every episode ends on the
        # stuck detector at exactly STUCK_WINDOW steps, and nothing is ever learned -- which is
        # precisely what the first run of this did. Normalized, only 23.5% land in the dead
        # zone. The real envelope still applies, just after scaling.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_spaces = {
            "encoders": spaces.Box(low=-1.0, high=1.0, shape=(OBS_HISTORY, 2), dtype=np.float32),
            # (forward, left, cos dtheta, sin dtheta). The heading goes in as a sin/cos pair so
            # there is no +/-180 deg discontinuity for the policy to trip over.
            "goal": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
        if include_lidar:
            obs_spaces["lidar"] = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_HISTORY, 3), dtype=np.float32
            )
        if include_image:
            obs_spaces["image"] = spaces.Box(
                low=0, high=255, shape=(CAM_RES, CAM_RES, 1), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)

        left_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_axle")
        right_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_axle")
        self._wheel_dof_adr = [self.model.jnt_dofadr[left_jid], self.model.jnt_dofadr[right_jid]]
        self._wheel_qpos_adr = [self.model.jnt_qposadr[left_jid], self.model.jnt_qposadr[right_jid]]
        self._wheel_geom_ids = [
            gi
            for gi in range(self.model.ngeom)
            if self.model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and self.model.geom_bodyid[gi] != 0
        ]
        self._friction_models = [motor.make_friction_model() for _ in self._wheel_dof_adr]

        self._obstacle_gids = [
            gi for gi in range(self.model.ngeom) if self.model.geom_group[gi] == 4
        ]
        self._obstacle_default_size = self.model.geom_size[self._obstacle_gids].copy()
        # Each obstacle geom hangs off its own mocap body (see scripts/gen_arena.py): moving
        # the body is what makes contacts follow it, so cache the mocap index per obstacle.
        self._obstacle_mocap_ids = [
            int(self.model.body_mocapid[self.model.geom_bodyid[gid]]) for gid in self._obstacle_gids
        ]
        assert all(mid >= 0 for mid in self._obstacle_mocap_ids), (
            "obstacles must be mocap bodies; re-run scripts/gen_arena.py"
        )
        # The arena is unwalled now, so an obstacle is the only thing that counts as a crash;
        # leaving the square is handled as its own out-of-bounds termination instead.
        self._hazard_gids = set(self._obstacle_gids)
        self._floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # One rangefinder per ray, grouped per beam; each fan folds to one distance by taking
        # the minimum, which is what a real cone-FOV ToF part reports.
        self._lidar_adr = np.array([
            [
                self.model.sensor_adr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"range_{beam}_{i}")
                ]
                for i in range(arena.LIDAR_RAYS_PER_BEAM)
            ]
            for beam in arena.LIDAR_BEAM_NAMES
        ])
        assert self._lidar_adr.min() >= 0, (
            "lidar sensors missing; re-run scripts/gen_rover_lidar.py"
        )

        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
        set_camera_tilt(self.model, "top_cam", CAMERA_ANGLE_DEG)
        self._default_cam_pos = self.model.cam_pos[self._cam_id].copy()
        self._default_cam_quat = self.model.cam_quat[self._cam_id].copy()
        self._default_light_diffuse = self.model.light_diffuse.copy()

        self._physics_dt = self.model.opt.timestep
        self._decimation = max(1, round(1.0 / (CONTROL_HZ * self._physics_dt)))
        self._episode_steps = 0

    def close(self):
        """Release the offscreen renderer's GL context -- see LineFollowerEnv.close() for why
        skipping this breaks every Renderer built after it in the same process."""
        if self.renderer is not None:
            self.renderer.close()
        super().close()

    # --- episode setup ---------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)

        level = CURRICULUM[self._level] if self.curriculum else CURRICULUM[-1]
        self._heading_tolerance = math.radians(level["heading_tol_deg"])

        start_xy, start_yaw = arena.sample_start(rng)
        # options={"goal": (forward, left)} or {"goal": (forward, left, heading_deg)} commands a
        # specific pose in the rover's own start frame -- how "go to (1, 2) heading North" is
        # issued from a script or the real-rover bridge, rather than letting the env pick.
        if options and "goal" in options:
            command = tuple(options["goal"])
            forward, left = command[0], command[1]
            self._goal_dheading = math.radians(command[2]) if len(command) > 2 else 0.0
            self._goal_xy = start_xy + np.array([
                forward * math.cos(start_yaw) - left * math.sin(start_yaw),
                forward * math.sin(start_yaw) + left * math.cos(start_yaw),
            ])
        else:
            self._goal_xy, goal_heading_world = arena.sample_pose_goal(
                rng, start_xy, *level["goal_distance"]
            )
            self._goal_dheading = wrap_angle(goal_heading_world - start_yaw)

        low, high = level["obstacles"]
        n_obstacles = int(rng.integers(low, high + 1))
        self._layout = self._place_obstacles(rng, start_xy, self._goal_xy, n_obstacles)

        self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] = start_xy[0], start_xy[1], 0.1
        # Robot forward is local +Y (see teleop_rover.py), so a world heading of start_yaw is a
        # rotation of start_yaw - 90 deg about z.
        half = (start_yaw - math.pi / 2) / 2
        self.data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]

        self._randomize(rng)
        mujoco.mj_forward(self.model, self.data)

        self._episode_steps = 0
        self._prev_wheel_angle = self.data.qpos[self._wheel_qpos_adr].copy()
        self._reset_action_buffer(rng)
        self._sensor_history.clear()
        self._recent_positions.clear()
        self._distance_at_window_start = np.inf

        # Odometry starts at the spawn pose by definition: where the rover thinks it is
        # relative to where it started, which is the frame the command is given in.
        self._odom = np.zeros(3)
        self._start_xy, self._start_yaw = start_xy, start_yaw
        self._goal_local_truth = self._world_to_start_frame(self._goal_xy)

        # Progress is measured along the shortest path *around* obstacles, not straight-line
        # distance -- the quantity A*/Dijkstra optimise. Straight-line distance rewards driving
        # into an obstacle; this does not. Built once here, read as a lookup per step.
        self._dist_field, self._field_axis, self._field_cell = arena.distance_field(
            self._layout, self._goal_xy
        )
        self._initial_distance = max(self._path_distance(start_xy), 1e-3)
        self._prev_distance = self._initial_distance

        self._lidar_reading = self._read_lidar(force=True)
        self._cached_image = self._capture_processed_image()

        return self._get_obs(), {
            "goal_command": np.append(self._goal_local_truth, math.degrees(self._goal_dheading)),
            "curriculum_level": self._level,
        }

    # --- stepping --------------------------------------------------------------------------

    def step(self, action):
        # [-1, 1] in, rad/s out. Noise is added after scaling because ACTION_NOISE_STD_RANGE is
        # in rad/s -- a physical quantity, not a fraction of the action range.
        action = np.clip(action, -1.0, 1.0) * arena.MAX_WHEEL_SPEED
        noisy = action + self.np_random.normal(0.0, self._action_noise_std, size=2)
        self._action_buffer.append(noisy.astype(np.float32))
        target_omega = self._apply_actuator_envelope(self._action_buffer[0])

        for _ in range(self._decimation):
            measured_omega = self.data.qvel[self._wheel_dof_adr]
            torque = motor.motor_torque(target_omega, measured_omega)
            self.data.ctrl[:] = torque
            for i, dof_adr in enumerate(self._wheel_dof_adr):
                motor.apply_friction(
                    self._friction_models[i], self.model, dof_adr, measured_omega[i]
                )
            mujoco.mj_step(self.model, self.data)
        self._episode_steps += 1

        self._cached_image = self._capture_processed_image()
        self._lidar_reading = self._read_lidar()
        obs = self._get_obs()  # also advances the odometry estimate from this step's ticks

        xy = self.data.qpos[:2].copy()
        if len(self._recent_positions) == 0:
            self._distance_at_window_start = self._prev_distance
        elif len(self._recent_positions) == STUCK_WINDOW:
            # The deque is about to drop its oldest sample, so remember the distance that
            # sample corresponds to as the new window baseline.
            self._distance_at_window_start = self._prev_distance
        self._recent_positions.append(xy)

        distance = self._path_distance(xy)
        heading_error = abs(self._heading_error())

        reward = PROGRESS_WEIGHT * (self._prev_distance - distance) / self._initial_distance
        reward -= STEP_PENALTY
        self._prev_distance = distance

        clearance = self._obstacle_clearance(xy)
        if clearance < PROXIMITY_THRESHOLD_M:
            reward -= PROXIMITY_WEIGHT * (1.0 - clearance / PROXIMITY_THRESHOLD_M)

        arrived = distance <= GOAL_TOLERANCE_M and heading_error <= self._heading_tolerance
        collided = self._collided()
        out_of_bounds = bool(np.any(np.abs(xy) > arena.ARENA_HALF + arena.OUT_OF_BOUNDS_TOLERANCE))
        stuck = self._is_stuck()
        tipped_over = self.data.qpos[2] < FALL_HEIGHT

        if arrived:
            # Guaranteed base plus an accuracy share, so a scruffier arrival earns less but is
            # never punished for having made it.
            position_score = 1.0 - min(distance / GOAL_TOLERANCE_M, 1.0)
            heading_score = 1.0 - min(heading_error / max(self._heading_tolerance, 1e-6), 1.0)
            reward += ARRIVAL_BASE + ARRIVAL_ACCURACY * 0.5 * (position_score + heading_score)
        if collided:
            reward -= COLLISION_PENALTY
        if out_of_bounds:
            reward -= OUT_OF_BOUNDS_PENALTY

        terminated = arrived or collided or out_of_bounds or stuck or tipped_over
        truncated = self._episode_steps >= MAX_EPISODE_STEPS
        if terminated or truncated:
            self._record_outcome(arrived)

        odom_distance = float(np.linalg.norm(self._odom[:2] - self._goal_local_truth))
        info = {
            "curriculum_level": self._level,
            "arrived": arrived,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "stuck": stuck,
            "distance": float(np.linalg.norm(self._goal_xy - xy)),
            "heading_error_deg": math.degrees(heading_error),
            # What the rover *believes*. On the real rover there is no ground truth, so it will
            # stop where this says it has arrived -- tracking both quantifies how much sim
            # success overstates real success.
            "odometry_distance": odom_distance,
            "arrived_per_odometry": odom_distance <= GOAL_TOLERANCE_M,
            "odometry_error": float(
                np.linalg.norm(self._odom[:2] - self._world_to_start_frame(xy))
            ),
        }
        return obs, reward, terminated, truncated, info

    # --- helpers ---------------------------------------------------------------------------

    def _apply_actuator_envelope(self, omega):
        """Clip to the motors' top speed and zero anything inside their dead zone. Below
        MIN_VELOCITY the real wheels cannot overcome friction at all, so a command in that band
        produces no motion -- the policy should learn that here rather than meet it for the
        first time on hardware."""
        omega = np.clip(omega, -self._max_omega, self._max_omega)
        return np.where(np.abs(omega) < self._min_omega, 0.0, omega)

    def _true_yaw(self):
        """World heading of the rover's forward (+Y local) axis."""
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self.data.qpos[3:7])
        forward = mat.reshape(3, 3) @ np.array([0.0, 1.0, 0.0])
        return math.atan2(forward[1], forward[0])

    def _heading_error(self):
        """Signed difference between the rover's true heading and the commanded one."""
        return wrap_angle(self._true_yaw() - (self._start_yaw + self._goal_dheading))

    def _path_distance(self, xy):
        """Geodesic distance to the goal around obstacles, from the precomputed field. Cells
        the rover cannot reach come back as inf; fall back to straight-line there rather than
        handing the reward an infinity."""
        i, j = self._field_cell(xy)
        value = self._dist_field[i, j]
        if not np.isfinite(value):
            return float(np.linalg.norm(self._goal_xy - np.asarray(xy)))
        return float(value)

    def _obstacle_clearance(self, xy):
        """True distance from the rover's centre to the nearest obstacle surface."""
        best = np.inf
        for centre, radius, is_box in self._layout:
            delta = np.abs(np.asarray(xy) - centre)
            if is_box:
                gap = float(np.linalg.norm(np.maximum(delta - radius, 0.0)))
            else:
                gap = float(max(np.linalg.norm(delta) - radius, 0.0))
            best = min(best, gap)
        return best

    def _is_stuck(self):
        """Going nowhere *and* getting no closer: wedged against something, or circling.

        Deliberately not just "did not translate". Arrival requires both position and heading,
        so a rover that has reached the goal has to rotate in place to align -- pure zero
        translation, and legitimate work. Testing progress as well as displacement separates
        the two; without this the detector fired mid-alignment and no episode could ever
        finish (measured: 64.7% of baseline episodes ended stuck, none arrived)."""
        if len(self._recent_positions) < STUCK_WINDOW:
            return False
        if self._prev_distance <= GOAL_TOLERANCE_M:
            return False  # at the goal, turning to face the commanded heading
        span = np.ptp(np.array(self._recent_positions), axis=0)
        if np.linalg.norm(span) >= STUCK_DISTANCE_M:
            return False
        # Also require that it has not been closing on the goal during the window.
        return bool(self._distance_at_window_start - self._prev_distance < STUCK_DISTANCE_M)

    def _record_outcome(self, arrived):
        """Promote this env to the next curriculum level once it arrives often enough. Each env
        keeps its own history: SubprocVecEnv gives no cheap way to share state, and letting
        workers advance independently keeps the batch a mix of difficulties."""
        if not self.curriculum:
            return
        self._recent_arrivals.append(bool(arrived))
        at_last_level = self._level >= len(CURRICULUM) - 1
        if at_last_level or len(self._recent_arrivals) < CURRICULUM_WINDOW:
            return
        if np.mean(self._recent_arrivals) >= CURRICULUM_PROMOTE_RATE:
            self._level += 1
            self._recent_arrivals.clear()

    def _world_to_start_frame(self, xy):
        """World (x, y) in the rover's spawn frame -- the frame the goal command and the
        odometry estimate both live in."""
        delta = np.asarray(xy, dtype=np.float64) - self._start_xy
        cos_y, sin_y = math.cos(self._start_yaw), math.sin(self._start_yaw)
        return np.array([
            delta[0] * cos_y + delta[1] * sin_y,
            -delta[0] * sin_y + delta[1] * cos_y,
        ])

    def _reset_action_buffer(self, rng):
        latency_steps = (
            int(rng.integers(ACTION_LATENCY_STEPS_RANGE[0], ACTION_LATENCY_STEPS_RANGE[1] + 1))
            if self.domain_randomize
            else 0
        )
        length = latency_steps + 1
        self._action_buffer = deque([np.zeros(2, dtype=np.float32)] * length, maxlen=length)

    def _place_obstacles(self, rng, start_xy, goal_xy, count):
        """Move `count` obstacles into the arena and park the rest outside it. Returns the
        layout as (centre, radius, is_box) so the reward and the distance field work off the
        same geometry the physics does."""
        layout = arena.sample_obstacles(rng, start_xy, goal_xy, count)
        park_x, park_y = arena.OBSTACLE_PARKING_XY
        # The pool alternates box/cylinder by slot, so taking slots 0..n-1 in order would make
        # the shape mix a fixed function of the count. Shuffling makes shapes random too.
        chosen = list(rng.permutation(len(self._obstacle_gids)))
        order = {slot: rank for rank, slot in enumerate(chosen)}
        placed = []
        for slot, gid in enumerate(self._obstacle_gids):
            mocap_id = self._obstacle_mocap_ids[slot]
            rank = order[slot]
            is_box = self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
            if rank < len(layout):
                xy, radius = layout[rank]
                self.data.mocap_pos[mocap_id] = [xy[0], xy[1], arena.OBSTACLE_HEIGHT / 2]
                size = self._obstacle_default_size[slot].copy()
                size[0] = radius
                if is_box:
                    size[1] = radius
                self.model.geom_size[gid] = size
                # geom_rbound is the compiled bounding-sphere radius the broadphase culls with;
                # resizing without it leaves the cull using the old size.
                self.model.geom_rbound[gid] = float(np.linalg.norm(size))
                placed.append((np.asarray(xy, dtype=float), float(radius), bool(is_box)))
            else:
                self.data.mocap_pos[mocap_id] = [
                    park_x + slot * 0.5,
                    park_y,
                    arena.OBSTACLE_HEIGHT / 2,
                ]
        return placed

    def _randomize(self, rng):
        """Per-episode domain randomization -- see the table in docs/rl-goal-nav.md."""
        if not self.domain_randomize:
            self._action_noise_std = 0.0
            self._encoder_noise_std = 0.0
            self._lidar_noise_std = 0.0
            self._brightness = 1.0
            self._white_balance = np.ones(3)
            self._pixel_noise_std = 0.0
            self._resolution_level = CAM_RES
            self._encoder_scale = np.ones(2)
            self._max_omega = arena.MAX_WHEEL_SPEED
            self._min_omega = arena.MIN_WHEEL_SPEED
            self.model.cam_fovy[self._cam_id] = 60.0
            for gid in self._wheel_geom_ids:
                self.model.geom_friction[gid, 0] = 1.0
            return

        self._action_noise_std = rng.uniform(*ACTION_NOISE_STD_RANGE)
        self._encoder_noise_std = rng.uniform(*ENCODER_NOISE_STD_RANGE)
        self._lidar_noise_std = rng.uniform(*LIDAR_NOISE_STD_RANGE)
        self._brightness = rng.uniform(*BRIGHTNESS_RANGE)
        self._white_balance = rng.uniform(*WHITE_BALANCE_RANGE, size=3)
        self._pixel_noise_std = rng.uniform(*PIXEL_NOISE_STD_RANGE)
        self._resolution_level = int(rng.choice(RESOLUTION_LEVELS))
        self.model.cam_fovy[self._cam_id] = rng.uniform(*FOVY_RANGE)

        # Systematic odometry error, sampled per wheel. This is the DR that decides whether
        # dead reckoning survives contact with reality: Gaussian tick noise averages out, but a
        # wheel 2% off nominal, or slipping on carpet, produces a steady curve whose error grows
        # without bound. Unequal left/right scales reproduce both a wheel-radius mismatch and an
        # effective track-width error, which are the two ways this fails on real hardware.
        slip = rng.uniform(*arena.WHEEL_SLIP_RANGE)
        self._encoder_scale = rng.uniform(*arena.WHEEL_RADIUS_SCALE_RANGE, size=2) * (1.0 + slip)
        self._max_omega = arena.MAX_WHEEL_SPEED * rng.uniform(*arena.MAX_SPEED_SCALE_RANGE)
        self._min_omega = arena.MIN_WHEEL_SPEED * rng.uniform(*arena.MIN_SPEED_SCALE_RANGE)

        wheel_friction = rng.uniform(*WHEEL_FRICTION_RANGE)
        for gid in self._wheel_geom_ids:
            self.model.geom_friction[gid, 0] = wheel_friction

        self.model.light_diffuse[:] = self._default_light_diffuse * rng.uniform(0.6, 1.3)
        pos_jitter = rng.uniform(-0.005, 0.005, size=3)
        self.model.cam_pos[self._cam_id] = self._default_cam_pos + pos_jitter
        angle_jitter = math.radians(rng.uniform(-3, 3))
        jitter_quat = np.array([math.cos(angle_jitter / 2), math.sin(angle_jitter / 2), 0.0, 0.0])
        mujoco.mju_mulQuat(self.model.cam_quat[self._cam_id], self._default_cam_quat, jitter_quat)

        # Obstacles are whatever happens to be lying around a classroom, not a fixed prop.
        for gid in self._obstacle_gids:
            self.model.geom_rgba[gid, :3] = rng.uniform(0.15, 0.75, size=3)

    def _collided(self):
        """True if any rover geom is touching an obstacle. The floor is excluded -- the rover
        is supposed to be touching that."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == self._floor_gid or g2 == self._floor_gid:
                continue
            if (g1 in self._hazard_gids) != (g2 in self._hazard_gids):
                return True
        return False

    def _read_lidar(self, force=False):
        """Three beam distances in metres, each the nearest return across its fan of rays.
        Held between refreshes to model the real poller's alternating queries."""
        if not force and self._episode_steps % LIDAR_DECIMATION != 0:
            return self._lidar_reading
        rays = np.array(self.data.sensordata[self._lidar_adr], dtype=np.float64)
        rays = np.where(rays < 0, arena.LIDAR_MAX_RANGE, rays)
        raw = rays.min(axis=1)
        raw = raw + self.np_random.normal(0.0, self._lidar_noise_std, size=3)
        raw = np.clip(raw, 0.0, arena.LIDAR_MAX_RANGE)
        return (np.round(raw / LIDAR_QUANTUM_M) * LIDAR_QUANTUM_M).astype(np.float32)

    def _capture_processed_image(self):
        """Same camera-realism pipeline as LineFollowerRealEnv._capture_processed_image()."""
        if not self.include_image:
            return None
        self.renderer.update_scene(self.data, camera="top_cam")
        rgb = self.renderer.render().astype(np.float32) * self._white_balance
        gray = rgb.mean(axis=-1, keepdims=True) * self._brightness
        gray += self.np_random.normal(0.0, self._pixel_noise_std, size=gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return _pixelate(gray, self._resolution_level, CAM_RES)

    def _get_obs(self):
        """Read the encoders, advance the dead-reckoned pose with them, and express the goal
        pose in that estimate's frame. The odometry consumes the same noisy, quantized,
        systematically-scaled ticks the policy sees, so the goal vector drifts the way it will
        on the real rover instead of tracking ground truth for free."""
        wheel_angle = self.data.qpos[self._wheel_qpos_adr]
        delta_angle = (wheel_angle - self._prev_wheel_angle) * self._encoder_scale
        self._prev_wheel_angle = wheel_angle.copy()

        noise_rad = self.np_random.normal(
            0.0, self._encoder_noise_std * self._physics_dt * self._decimation, size=2
        )
        ticks = np.round((delta_angle + noise_rad) * ENCODER_CPR_WHEEL / (2 * np.pi))

        self._odom = arena.integrate_odometry(self._odom, ticks[0], ticks[1], ENCODER_CPR_WHEEL)
        goal_local = arena.goal_in_body_frame(self._odom, self._goal_local_truth)
        heading_to_go = wrap_angle(self._goal_dheading - self._odom[2])

        self._sensor_history.appendleft((
            (ticks / ENCODER_SCALE).astype(np.float32),
            (self._lidar_reading / LIDAR_SCALE).astype(np.float32),
        ))
        # Before OBS_HISTORY readings exist the oldest is repeated, so the shape is fixed and
        # the padding is a plausible past rather than zeros.
        frames = list(self._sensor_history)
        frames += [frames[-1]] * (OBS_HISTORY - len(frames))

        obs = {
            "encoders": np.stack([f[0] for f in frames]),
            "goal": np.array(
                [
                    goal_local[0] / GOAL_SCALE,
                    goal_local[1] / GOAL_SCALE,
                    math.cos(heading_to_go),
                    math.sin(heading_to_go),
                ],
                dtype=np.float32,
            ),
        }
        if self.include_lidar:
            obs["lidar"] = np.stack([f[1] for f in frames])
        if self.include_image:
            obs["image"] = self._cached_image
        return obs
