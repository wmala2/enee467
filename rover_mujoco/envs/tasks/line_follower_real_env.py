from collections import deque

from gymnasium import spaces
import mujoco
import numpy as np

from envs import motor
from envs.camera import onboard_scene_option
from envs.tasks.line_follower_env import CAM_RES
from envs.tasks.line_follower_env import CENTER_WEIGHT
from envs.tasks.line_follower_env import CONTROL_HZ
from envs.tasks.line_follower_env import FALL_HEIGHT
from envs.tasks.line_follower_env import LineFollowerEnv
from envs.tasks.line_follower_env import MAX_EPISODE_STEPS
from envs.tasks.line_follower_env import MAX_LINE_LOST_STEPS
from envs.tracks import oval_waypoints
from envs.tracks import s_curve_waypoints

# --- Domain randomization ranges ---
# Starting points, not measured hardware values — see the DR table in
# docs/rl-line-follower.md for the reasoning behind each one and what still needs real
# characterization (motor curves fitted via BAM, per-surface friction, camera datasheet specs).
ACTION_NOISE_STD_RANGE = (0.0, 0.3)  # rad/s: per-step Gaussian jitter on commanded speed
ACTION_LATENCY_STEPS_RANGE = (0, 2)  # control-loop steps of command delay (@ CONTROL_HZ,
# so 0-200ms — real network/serial round-trip scale)
WHEEL_FRICTION_RANGE = (0.4, 1.2)  # floor-contact sliding friction, stands in for surface
ENCODER_NOISE_STD_RANGE = (0.0, 0.1)  # rad/s-equivalent: pre-quantization jitter on the
# angle reading (stray/missed quadrature edges), not
# noise on a velocity — see ENCODER_CPR_WHEEL below

# The real rover has no velocity sensor: its "e" UDP command (see wmala2/rover-firmware,
# src/control_server.cpp) returns raw cumulative quadrature counts, `long`, never reset on
# read — a *position* reading, not velocity. Any "velocity" is something a client derives
# by polling twice and differencing, which is exactly what this env now does (delta ticks
# since the last control step), instead of the idealized continuous rad/s this used to hand
# the policy directly (a real sim-to-real gap: no such clean number exists on the hardware).
# 680 counts/wheel-revolution is the firmware's own default (hardware_config.h,
# MOTOR_PROFILE=500, ENCODER_CPR_WHEEL) — not a guess, unlike most of this file's DR ranges.
ENCODER_CPR_WHEEL = 680.0
# Generous bound on ticks/control-step: the action space alone (±10 rad/s, CONTROL_HZ=10Hz)
# already implies up to ~108 ticks/step; this leaves headroom for transient overshoot.
ENCODER_TICKS_BOUND = 200.0

# The camera capture rate is the same real constraint that motivates CONTROL_HZ (LineFollowerEnv's
# control-decimation): the rover's \capture endpoint and its command loop both run at ~10Hz, so
# one control decision = one fresh camera frame here, rather than tracking two independent rates.
CAMERA_RATE_HZ = CONTROL_HZ

# Camera-realism DR: everything downstream of "the pixels the rover's own camera hardware
# would actually produce," as opposed to LineFollowerEnv's floor/line color + camera mount
# pose. CAM_RES itself can't be randomized directly (the observation_space shape must stay
# fixed for SB3), so "resolution" is approximated by pixelating a fixed-size render.
FOVY_RANGE = (50.0, 70.0)  # deg: lens FOV manufacturing/mounting tolerance
BRIGHTNESS_RANGE = (0.6, 1.4)  # exposure/ambient-light multiplier on pixel value
WHITE_BALANCE_RANGE = (0.85, 1.15)  # per-RGB-channel gain before grayscale conversion
PIXEL_NOISE_STD_RANGE = (0.0, 15.0)  # Gaussian sensor/JPEG-quality noise (0-255 scale)
RESOLUTION_LEVELS = (16, 32, 64)  # effective sensor resolution; must all divide CAM_RES


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
    CAMERA_RATE_HZ (not every physics step), plus left/right wheel encoder ticks accrued
    since the last control step (quantized, noisy — matching the real rover's raw quadrature
    count query, not an idealized velocity; see ENCODER_CPR_WHEEL) — no ground-truth
    position, and no privileged image update rate either.

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
            "encoders": spaces.Box(
                low=-ENCODER_TICKS_BOUND, high=ENCODER_TICKS_BOUND, shape=(2,), dtype=np.float32
            ),
        })

        left_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_axle")
        right_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_axle")
        self._wheel_dof_adr = [self.model.jnt_dofadr[left_jid], self.model.jnt_dofadr[right_jid]]
        # Wheel angle (qpos, not qvel) — a hinge joint's own single qpos entry — is what the
        # encoder-ticks observation is derived from, matching the real quadrature counter.
        self._wheel_qpos_adr = [self.model.jnt_qposadr[left_jid], self.model.jnt_qposadr[right_jid]]

        # Wheel cylinder collision geoms: what DR varies to stand in for different floor
        # materials (grass/concrete/carpet), since contact friction is the wheel-geom/floor-geom
        # pair — varying the wheel side is the practical way to sweep effective friction in MuJoCo.
        # Distinct from the motor's own internal (gearbox/brush) friction below.
        self._wheel_geom_ids = [
            gi
            for gi in range(self.model.ngeom)
            if self.model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]

        # One BAM friction model per wheel (Stribeck, JGA25-371-scaled — see envs/motor.py)
        self._friction_models = [motor.make_friction_model() for _ in self._wheel_dof_adr]

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        rng = self.np_random

        if self.domain_randomize:
            self._action_noise_std = rng.uniform(*ACTION_NOISE_STD_RANGE)
            self._encoder_noise_std = rng.uniform(*ENCODER_NOISE_STD_RANGE)
            latency_steps = int(
                rng.integers(ACTION_LATENCY_STEPS_RANGE[0], ACTION_LATENCY_STEPS_RANGE[1] + 1)
            )
        else:
            self._action_noise_std = 0.0
            self._encoder_noise_std = 0.0
            latency_steps = 0
        # Ticks accrue from here — matches the real rover's "r" (reset encoders) convention.
        self._prev_wheel_angle = self.data.qpos[self._wheel_qpos_adr].copy()
        buffer_len = (
            latency_steps + 1
        )  # +1 so index [0] is exactly latency_steps old, not latency_steps-1
        self._action_buffer = deque([np.zeros(2, dtype=np.float32)] * buffer_len, maxlen=buffer_len)

        # This whole block used to run unconditionally, ignoring domain_randomize=False
        # (a real bug: that flag only ever gated the base class's floor/line/camera-pose
        # jitter, not this subclass's own motor/camera-realism DR) — fixed so "DR off"
        # actually means off, for clean single-variable eval/debugging.
        if self.domain_randomize:
            wheel_friction = rng.uniform(*WHEEL_FRICTION_RANGE)
            self.model.cam_fovy[self._cam_id] = rng.uniform(*FOVY_RANGE)
            self._brightness = rng.uniform(*BRIGHTNESS_RANGE)
            self._white_balance = rng.uniform(*WHITE_BALANCE_RANGE, size=3)
            self._pixel_noise_std = rng.uniform(*PIXEL_NOISE_STD_RANGE)
            self._resolution_level = int(rng.choice(RESOLUTION_LEVELS))
        else:
            wheel_friction = 1.0
            self.model.cam_fovy[self._cam_id] = 60.0  # rover.xml's own top_cam default
            self._brightness = 1.0
            self._white_balance = np.ones(3)
            self._pixel_noise_std = 0.0
            self._resolution_level = CAM_RES  # full resolution, no pixelation
        for gid in self._wheel_geom_ids:
            self.model.geom_friction[gid, 0] = wheel_friction

        self._cached_image = self._capture_processed_image()

        return self._get_real_obs(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        noisy_action = action + self.np_random.normal(0.0, self._action_noise_std, size=2)

        # Action latency: queue the noisy command, apply whatever's aged out the other end.
        # The target speed is held fixed for the whole decimation window below — matching
        # CONTROL_HZ, the real command rate, rather than letting the policy whipsaw the
        # target every 0.002s physics tick (see CONTROL_HZ's comment in line_follower_env.py
        # for the degenerate "vibrate in place" policy that let the motor's own feedback
        # loop exploit).
        self._action_buffer.append(noisy_action.astype(np.float32))
        target_omega = self._action_buffer[0]

        # Motor model: DC-motor torque toward the (delayed, noisy) target speed, plus this
        # wheel's own BAM Stribeck friction — replaces MuJoCo's built-in velocity servo.
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

        # One fresh camera frame per control decision — CAMERA_RATE_HZ == CONTROL_HZ, so
        # this coincides exactly with the decimation window above rather than needing its
        # own separate counter.
        self._cached_image = self._capture_processed_image()

        obs = self._get_real_obs()
        error = self._line_error(self._cached_image)
        progress_reward, finished = self._track_progress_reward()

        if error is None:
            self._lost_steps += 1
            reward = -1.0
        else:
            self._lost_steps = 0
            reward = CENTER_WEIGHT * (1.0 - abs(error))
        reward += progress_reward

        tipped_over = self.data.qpos[2] < FALL_HEIGHT
        terminated = tipped_over or finished
        truncated = (
            self._lost_steps >= MAX_LINE_LOST_STEPS or self._episode_steps >= MAX_EPISODE_STEPS
        )
        return obs, reward, terminated, truncated, {}

    def _capture_processed_image(self):
        """Render one frame and run it through this episode's camera-realism DR: white
        balance (per-channel gain, applied before collapsing to grayscale since that's
        where a real color-balance shift would actually act), brightness, sensor noise
        (a stand-in for JPEG-quality artifacts — real compression noise is structured
        block artifacts, not i.i.d. Gaussian, but this is far simpler and still forces
        the policy not to trust exact pixel values), and effective resolution."""
        self.renderer.update_scene(self.data, camera="top_cam", scene_option=onboard_scene_option())
        rgb = self.renderer.render().astype(np.float32) * self._white_balance
        gray = rgb.mean(axis=-1, keepdims=True) * self._brightness
        gray += self.np_random.normal(0.0, self._pixel_noise_std, size=gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return _pixelate(gray, self._resolution_level, CAM_RES)

    def _get_real_obs(self):
        """Left/right encoder ticks accrued since the last control step — the real rover's
        actual sensor primitive (see ENCODER_CPR_WHEEL above), not a velocity: this is what
        a client gets by polling the "e" UDP command twice and differencing. Angle-domain
        noise (stray/missed quadrature edges) is applied before quantizing to integer ticks,
        so the returned value has the same discreteness a real read would."""
        wheel_angle = self.data.qpos[self._wheel_qpos_adr]
        delta_angle = wheel_angle - self._prev_wheel_angle
        self._prev_wheel_angle = wheel_angle.copy()

        noise_rad = self.np_random.normal(
            0.0, self._encoder_noise_std * self._physics_dt * self._decimation, size=2
        )
        ticks = np.round((delta_angle + noise_rad) * ENCODER_CPR_WHEEL / (2 * np.pi))
        return {"image": self._cached_image, "encoders": ticks.astype(np.float32)}
