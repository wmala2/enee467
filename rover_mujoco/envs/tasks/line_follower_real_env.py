from collections import deque

import numpy as np
import mujoco
from gymnasium import spaces

from envs import motor
from envs.tracks import oval_waypoints, s_curve_waypoints
from envs.tasks.line_follower_env import (
    LineFollowerEnv,
    CAM_RES,
    FALL_HEIGHT,
    MAX_LINE_LOST_STEPS,
    MAX_EPISODE_STEPS,
)

# --- Domain randomization ranges ---
# Starting points, not measured hardware values — see the DR table in
# docs/rl-line-follower.md for the reasoning behind each one and what still needs real
# characterization (motor curves fitted via BAM, per-surface friction, camera datasheet specs).
ACTION_NOISE_STD_RANGE = (0.0, 0.3)    # rad/s: per-step Gaussian jitter on commanded speed
ACTION_LATENCY_STEPS_RANGE = (0, 2)    # control-loop steps of command delay
WHEEL_FRICTION_RANGE = (0.4, 1.2)      # floor-contact sliding friction, stands in for surface
ENCODER_NOISE_STD_RANGE = (0.0, 0.1)   # rad/s: measurement noise on the encoder reading
CAMERA_RATE_HZ = 10.0                  # matches the real rover's \capture endpoint rate

# Camera-realism DR: everything downstream of "the pixels the rover's own camera hardware
# would actually produce," as opposed to LineFollowerEnv's floor/line color + camera mount
# pose. CAM_RES itself can't be randomized directly (the observation_space shape must stay
# fixed for SB3), so "resolution" is approximated by pixelating a fixed-size render.
FOVY_RANGE = (50.0, 70.0)              # deg: lens FOV manufacturing/mounting tolerance
BRIGHTNESS_RANGE = (0.6, 1.4)          # exposure/ambient-light multiplier on pixel value
WHITE_BALANCE_RANGE = (0.85, 1.15)     # per-RGB-channel gain before grayscale conversion
PIXEL_NOISE_STD_RANGE = (0.0, 15.0)    # Gaussian sensor/JPEG-quality noise (0-255 scale)
RESOLUTION_LEVELS = (16, 32, 64)       # effective sensor resolution; must all divide CAM_RES


def _pixelate(img, level, full_res):
    """Block-average `img` down to `level`x`level` then nearest-neighbor back up to
    `full_res`x`full_res`, approximating a lower-resolution sensor while keeping the
    observation's actual array shape fixed. `level` must evenly divide `full_res`."""
    if level >= full_res:
        return img
    factor = full_res // level
    small = img.reshape(level, factor, level, factor, 1).mean(axis=(1, 3))
    return np.repeat(np.repeat(small, factor, axis=0), factor, axis=1).astype(np.uint8)

# Torque-actuated scene variants (see assets/robots/rover/*_real.xml) — the wheel joints
# take raw torque (Nm) here, computed each step by envs/motor.py's JGA25-371 model, rather
# than a MuJoCo <velocity> servo doing that internally like LineFollower-v0 uses.
TRACKS_REAL = {
    "rover_line_oval_real.xml": oval_waypoints,
    "rover_line_s_curve_real.xml": s_curve_waypoints,
}


class LineFollowerRealEnv(LineFollowerEnv):
    """LineFollowerEnv, constrained to what the real rover can actually sense and command,
    plus domain randomization over the physical properties sim can't otherwise get right.

    Observation: the same onboard camera image as LineFollowerEnv, refreshed only at
    CAMERA_RATE_HZ (not every physics step), plus noisy left/right wheel encoder readings
    (angular velocity) — no ground-truth position, and no privileged image update rate either.

    Action: left/right wheel velocity commands, same shape/units as LineFollowerEnv and
    teleop_rover.py. This is an assumption pending the real rover-firmware repo's exact
    command format (units, scaling, PWM vs. velocity) — reconcile it once that's available;
    everything else in this env (motor model, latency, noise) layers on top of whatever the
    real command interface turns out to be.

    Wheel torque comes from envs/motor.py's JGA25-371 DC-motor model (BAM friction + a
    hand-derived electrical model from the datasheet) instead of MuJoCo's built-in velocity
    servo, so this env's scenes (assets/robots/rover/*_real.xml) use raw torque actuators.
    """

    TRACKS = TRACKS_REAL

    def __init__(self, render_mode=None, domain_randomize=True):
        super().__init__(render_mode=render_mode, domain_randomize=domain_randomize)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(CAM_RES, CAM_RES, 1), dtype=np.uint8),
            "encoders": spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
        })

        left_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_axle")
        right_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_axle")
        self._wheel_dof_adr = [self.model.jnt_dofadr[left_jid], self.model.jnt_dofadr[right_jid]]

        # Wheel cylinder collision geoms: what DR varies to stand in for different floor
        # materials (grass/concrete/carpet), since contact friction is the wheel-geom/floor-geom
        # pair — varying the wheel side is the practical way to sweep effective friction in MuJoCo.
        # Distinct from the motor's own internal (gearbox/brush) friction below.
        self._wheel_geom_ids = [
            gi for gi in range(self.model.ngeom)
            if self.model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]

        # One BAM friction model per wheel (Stribeck, JGA25-371-scaled — see envs/motor.py)
        self._friction_models = [motor.make_friction_model() for _ in self._wheel_dof_adr]

        self._physics_dt = self.model.opt.timestep
        self._capture_period = max(1, round(1.0 / (CAMERA_RATE_HZ * self._physics_dt)))

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        rng = self.np_random

        self._action_noise_std = rng.uniform(*ACTION_NOISE_STD_RANGE)
        self._encoder_noise_std = rng.uniform(*ENCODER_NOISE_STD_RANGE)
        latency_steps = int(rng.integers(ACTION_LATENCY_STEPS_RANGE[0], ACTION_LATENCY_STEPS_RANGE[1] + 1))
        buffer_len = latency_steps + 1  # +1 so index [0] is exactly latency_steps old, not latency_steps-1
        self._action_buffer = deque([np.zeros(2, dtype=np.float32)] * buffer_len, maxlen=buffer_len)

        wheel_friction = rng.uniform(*WHEEL_FRICTION_RANGE)
        for gid in self._wheel_geom_ids:
            self.model.geom_friction[gid, 0] = wheel_friction

        # Camera realism, redrawn each episode (see docs/rl-line-follower.md's DR table)
        self.model.cam_fovy[self._cam_id] = rng.uniform(*FOVY_RANGE)
        self._brightness = rng.uniform(*BRIGHTNESS_RANGE)
        self._white_balance = rng.uniform(*WHITE_BALANCE_RANGE, size=3)
        self._pixel_noise_std = rng.uniform(*PIXEL_NOISE_STD_RANGE)
        self._resolution_level = int(rng.choice(RESOLUTION_LEVELS))

        self._capture_counter = 0
        self._cached_image = self._capture_processed_image()

        return self._get_real_obs(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        noisy_action = action + self.np_random.normal(0.0, self._action_noise_std, size=2)

        # Action latency: queue the noisy command, apply whatever's aged out the other end
        self._action_buffer.append(noisy_action.astype(np.float32))
        target_omega = self._action_buffer[0]

        # Motor model: DC-motor torque toward the (delayed, noisy) target speed, plus this
        # wheel's own BAM Stribeck friction — replaces MuJoCo's built-in velocity servo.
        measured_omega = self.data.qvel[self._wheel_dof_adr]
        torque = motor.motor_torque(target_omega, measured_omega)
        self.data.ctrl[:] = torque
        for i, dof_adr in enumerate(self._wheel_dof_adr):
            motor.apply_friction(self._friction_models[i], self.model, dof_adr, measured_omega[i])

        mujoco.mj_step(self.model, self.data)
        self._episode_steps += 1

        self._capture_counter += 1
        if self._capture_counter >= self._capture_period:
            self._capture_counter = 0
            self._cached_image = self._capture_processed_image()
        # else: camera hasn't captured a new frame yet at CAMERA_RATE_HZ, reuse the last one

        obs = self._get_real_obs()
        error = self._line_error(self._cached_image)

        if error is None:
            self._lost_steps += 1
            reward = -1.0
        else:
            self._lost_steps = 0
            reward = 1.0 - abs(error)

        tipped_over = self.data.qpos[2] < FALL_HEIGHT
        terminated = tipped_over
        truncated = (
            self._lost_steps >= MAX_LINE_LOST_STEPS
            or self._episode_steps >= MAX_EPISODE_STEPS
        )
        return obs, reward, terminated, truncated, {}

    def _capture_processed_image(self):
        """Render one frame and run it through this episode's camera-realism DR: white
        balance (per-channel gain, applied before collapsing to grayscale since that's
        where a real color-balance shift would actually act), brightness, sensor noise
        (a stand-in for JPEG-quality artifacts — real compression noise is structured
        block artifacts, not i.i.d. Gaussian, but this is far simpler and still forces
        the policy not to trust exact pixel values), and effective resolution."""
        self.renderer.update_scene(self.data, camera="top_cam")
        rgb = self.renderer.render().astype(np.float32) * self._white_balance
        gray = rgb.mean(axis=-1, keepdims=True) * self._brightness
        gray += self.np_random.normal(0.0, self._pixel_noise_std, size=gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return _pixelate(gray, self._resolution_level, CAM_RES)

    def _get_real_obs(self):
        encoders = np.array(
            [self.data.qvel[adr] for adr in self._wheel_dof_adr], dtype=np.float32
        )
        encoders += self.np_random.normal(0.0, self._encoder_noise_std, size=2).astype(np.float32)
        return {"image": self._cached_image, "encoders": encoders}
