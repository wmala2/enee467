"""Goal navigation with obstacle avoidance, on the same sim-to-real footing as
LineFollowerRealEnv: drive to a commanded (x, y) offset inside a 5x5 m arena without hitting
anything, using only what the physical rover can actually sense.

The command is a coordinate, not something visible -- "drive to (1, 2)" means 1 m to the
rover's right and 2 m forward of wherever it started. Nothing in the world marks that spot,
so the only way to find it is to dead-reckon from the wheel encoders, which is exactly the
point: the goal vector the policy is handed drifts, because it is integrated from the same
noisy quantized ticks the policy sees. Obstacles come from the three-beam lidar (and, weakly,
the camera). See docs/rl-goal-nav.md.
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

# The line follower's 15.2 deg mount (nearly straight down) cannot see an obstacle at all --
# measured 0% changed pixels with a box 0.4 m ahead. At 60 deg the same box covers ~8% of the
# frame at 0.4 m but only ~1% at 0.8 m, so the camera is a short-range confirmation at best
# and the lidar is the real obstacle sensor here. Tune this if the mount changes.
CAMERA_ANGLE_DEG = 60.0

MAX_EPISODE_STEPS = 400  # control steps @ CONTROL_HZ = 40 simulated seconds
GOAL_TOLERANCE_M = 0.25  # how close counts as "arrived"

# Reward weights, deliberately shaped like LineFollowerEnv's: a dominant progress term
# normalized so a complete run sums to ~PROGRESS_WEIGHT no matter how far away the goal was
# sampled, plus a flat bonus for actually arriving.
PROGRESS_WEIGHT = 100.0
SUCCESS_BONUS = 50.0
COLLISION_PENALTY = 25.0  # one-off, and the episode ends there
STEP_PENALTY = 0.05  # per step, so dawdling costs something and stalling is never optimal

# Lidar realism. The real sensor answers in integer millimetres and the firmware reports a
# per-reading validity flag; MuJoCo's rangefinder returns -1 when the beam hits nothing,
# which we fold into the same "nothing within range" value.
LIDAR_NOISE_STD_RANGE = (0.0, 0.02)  # m: per-beam Gaussian jitter
LIDAR_QUANTUM_M = 0.001  # the sensor's millimetre resolution
# rover_control/encoder_poller.py alternates its "e" and "l" UDP queries, so each sensor
# refreshes at half the poll rate. rl_rover.py already polls encoders at the full rate for
# the policy, which leaves the lidar at every other control step -- modelled here by holding
# the previous reading, so the policy trains against the staleness it will actually get.
LIDAR_DECIMATION = 2


class GoalNavEnv(gym.Env):
    """Drive to a commanded offset in a 5x5 m arena without hitting the walls or the
    obstacles scattered around it.

    Observation (all of it available on the real rover):
      image     -- the same 64x64 grayscale onboard camera as LineFollowerRealEnv, through
                   the same camera-realism randomization, tilted forward to see obstacles.
      encoders  -- left/right wheel ticks accrued since the last control step.
      lidar     -- left/center/right beam distances in metres, clipped to LIDAR_MAX_RANGE,
                   quantized to millimetres, refreshed every LIDAR_DECIMATION steps.
      goal      -- (forward, left) metres to the goal in the rover's *estimated* body frame,
                   dead-reckoned from the noisy encoder ticks above. Never ground truth: this
                   drifts, and recovering from that drift is part of the task.

    Action: left/right wheel velocity targets, identical in shape and units to
    LineFollowerRealEnv, driven through envs/motor.py's JGA25-371 model.

    Reward uses ground-truth position (progress toward the goal, arrival, collisions) --
    the same reward-only privilege LineFollowerEnv documents: reward does not exist at
    deployment, the observation does.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, domain_randomize=True):
        super().__init__()
        self.domain_randomize = domain_randomize
        self.render_mode = render_mode

        model_path = os.path.join(
            os.path.dirname(__file__), "../../assets/robots/rover", SCENE_FILE
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=CAM_RES, width=CAM_RES)

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        # The goal bound is the arena diagonal plus headroom, since odometry drift can put the
        # estimate outside the walls.
        goal_bound = np.float32(arena.ARENA_SIZE * 1.5)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(CAM_RES, CAM_RES, 1), dtype=np.uint8),
            "encoders": spaces.Box(
                low=-ENCODER_TICKS_BOUND, high=ENCODER_TICKS_BOUND, shape=(2,), dtype=np.float32
            ),
            "lidar": spaces.Box(low=0.0, high=arena.LIDAR_MAX_RANGE, shape=(3,), dtype=np.float32),
            "goal": spaces.Box(low=-goal_bound, high=goal_bound, shape=(2,), dtype=np.float32),
        })

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
        wall_names = ("wall_north", "wall_south", "wall_east", "wall_west")
        self._wall_gids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in wall_names
        ]
        # Anything in here counts as "the rover hit something" when it shows up in a contact.
        self._hazard_gids = set(self._obstacle_gids) | set(self._wall_gids)
        self._floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self._lidar_adr = [
            self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)]
            for name in ("range_left", "range_center", "range_right")
        ]

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
        self.renderer.close()
        super().close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)

        start_xy, start_yaw = arena.sample_start(rng)
        # `options={"goal": (forward, left)}` commands a specific offset in the rover's own
        # starting frame -- this is how "drive to (1, 2)" is issued from a script or from the
        # real-rover bridge, rather than letting the env pick.
        if options and "goal" in options:
            forward, left = options["goal"]
            self._goal_xy = start_xy + np.array([
                forward * math.cos(start_yaw) - left * math.sin(start_yaw),
                forward * math.sin(start_yaw) + left * math.cos(start_yaw),
            ])
        else:
            self._goal_xy = arena.sample_goal(rng, start_xy)

        n_obstacles = int(rng.integers(3, arena.MAX_OBSTACLES + 1))
        self._place_obstacles(rng, start_xy, self._goal_xy, n_obstacles)

        self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] = start_xy[0], start_xy[1], 0.1
        # Robot forward is local +Y (see teleop_rover.py), so a heading of `start_yaw` in world
        # terms is a rotation of start_yaw - 90 deg about z.
        half = (start_yaw - math.pi / 2) / 2
        self.data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]

        self._randomize(rng)
        mujoco.mj_forward(self.model, self.data)

        self._episode_steps = 0
        self._prev_wheel_angle = self.data.qpos[self._wheel_qpos_adr].copy()
        self._reset_action_buffer(rng)

        # Odometry starts at the spawn pose by definition: the rover's estimate of where it is
        # relative to where it started, which is exactly the frame the command is given in.
        self._odom = np.zeros(3)
        self._start_xy, self._start_yaw = start_xy, start_yaw
        self._goal_local_truth = self._world_to_start_frame(self._goal_xy)

        self._initial_distance = float(np.linalg.norm(self._goal_xy - start_xy))
        self._prev_distance = self._initial_distance
        self._lidar_reading = self._read_lidar(force=True)
        self._cached_image = self._capture_processed_image()

        return self._get_obs(), {"goal_command": self._goal_local_truth}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        noisy_action = action + self.np_random.normal(0.0, self._action_noise_std, size=2)
        self._action_buffer.append(noisy_action.astype(np.float32))
        target_omega = self._action_buffer[0]

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

        distance = float(np.linalg.norm(self._goal_xy - self.data.qpos[:2]))
        reward = PROGRESS_WEIGHT * (self._prev_distance - distance) / self._initial_distance
        reward -= STEP_PENALTY
        self._prev_distance = distance

        arrived = distance <= GOAL_TOLERANCE_M
        collided = self._collided()
        tipped_over = self.data.qpos[2] < FALL_HEIGHT
        if arrived:
            reward += SUCCESS_BONUS
        if collided:
            reward -= COLLISION_PENALTY

        terminated = arrived or collided or tipped_over
        truncated = self._episode_steps >= MAX_EPISODE_STEPS
        info = {
            "arrived": arrived,
            "collided": collided,
            "distance": distance,
            "odometry_error": float(
                np.linalg.norm(self._odom[:2] - self._world_to_start_frame(self.data.qpos[:2]))
            ),
        }
        return obs, reward, terminated, truncated, info

    def _world_to_start_frame(self, xy):
        """World (x, y) expressed in the rover's spawn frame -- the frame the goal command and
        the odometry estimate both live in."""
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
        """Move `count` obstacles from the pool into the arena and park the rest outside the
        walls. MuJoCo compiles geometry once, so an episode-varying layout has to come from
        mutating model.geom_pos/geom_size, the same runtime trick LineFollowerEnv's appearance
        randomization uses."""
        layout = arena.sample_obstacles(rng, start_xy, goal_xy, count)
        park_x, park_y = arena.OBSTACLE_PARKING_XY
        for slot, gid in enumerate(self._obstacle_gids):
            if slot < len(layout):
                xy, radius = layout[slot]
                self.model.geom_pos[gid] = [xy[0], xy[1], arena.OBSTACLE_HEIGHT / 2]
                size = self._obstacle_default_size[slot].copy()
                size[0] = radius
                if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                    size[1] = radius
                self.model.geom_size[gid] = size
            else:
                self.model.geom_pos[gid] = [park_x + slot * 0.5, park_y, arena.OBSTACLE_HEIGHT / 2]

    def _randomize(self, rng):
        """Per-episode domain randomization. Deliberately the same knobs and ranges as
        LineFollowerRealEnv (motor lag, action noise/latency, encoder noise, wheel friction,
        camera FOV/brightness/white balance/pixel noise/resolution) so a policy trained here
        and a line-following policy have transferred across the same variation, plus the two
        this task adds: lidar noise and obstacle colour."""
        if not self.domain_randomize:
            self._action_noise_std = 0.0
            self._encoder_noise_std = 0.0
            self._lidar_noise_std = 0.0
            self._brightness = 1.0
            self._white_balance = np.ones(3)
            self._pixel_noise_std = 0.0
            self._resolution_level = CAM_RES
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
        """True if any rover geom is touching a wall or an obstacle. The floor is excluded --
        the rover is supposed to be touching that."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == self._floor_gid or g2 == self._floor_gid:
                continue
            if (g1 in self._hazard_gids) != (g2 in self._hazard_gids):
                return True
        return False

    def _read_lidar(self, force=False):
        """Three beam distances in metres. Held between refreshes to model the real poller's
        alternating encoder/lidar queries (see LIDAR_DECIMATION)."""
        if not force and self._episode_steps % LIDAR_DECIMATION != 0:
            return self._lidar_reading
        raw = np.array(self.data.sensordata[self._lidar_adr], dtype=np.float64)
        # MuJoCo reports -1 for "the beam hit nothing"; so does anything past our stated range.
        raw = np.where(raw < 0, arena.LIDAR_MAX_RANGE, raw)
        raw = raw + self.np_random.normal(0.0, self._lidar_noise_std, size=3)
        raw = np.clip(raw, 0.0, arena.LIDAR_MAX_RANGE)
        return (np.round(raw / LIDAR_QUANTUM_M) * LIDAR_QUANTUM_M).astype(np.float32)

    def _capture_processed_image(self):
        """Same camera-realism pipeline as LineFollowerRealEnv._capture_processed_image()."""
        self.renderer.update_scene(self.data, camera="top_cam")
        rgb = self.renderer.render().astype(np.float32) * self._white_balance
        gray = rgb.mean(axis=-1, keepdims=True) * self._brightness
        gray += self.np_random.normal(0.0, self._pixel_noise_std, size=gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return _pixelate(gray, self._resolution_level, CAM_RES)

    def _get_obs(self):
        """Read the encoders, advance the dead-reckoned pose with them, and express the goal
        in that estimate's frame. The odometry deliberately consumes the *noisy, quantized*
        ticks -- the same numbers handed to the policy -- so the goal vector drifts the way it
        would on the real rover instead of tracking ground truth for free."""
        wheel_angle = self.data.qpos[self._wheel_qpos_adr]
        delta_angle = wheel_angle - self._prev_wheel_angle
        self._prev_wheel_angle = wheel_angle.copy()

        noise_rad = self.np_random.normal(
            0.0, self._encoder_noise_std * self._physics_dt * self._decimation, size=2
        )
        ticks = np.round((delta_angle + noise_rad) * ENCODER_CPR_WHEEL / (2 * np.pi))

        self._odom = arena.integrate_odometry(self._odom, ticks[0], ticks[1], ENCODER_CPR_WHEEL)
        goal_local = arena.goal_in_body_frame(self._odom, self._goal_local_truth)

        return {
            "image": self._cached_image,
            "encoders": ticks.astype(np.float32),
            "lidar": self._lidar_reading,
            "goal": goal_local,
        }
