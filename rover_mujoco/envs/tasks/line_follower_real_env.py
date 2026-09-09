from collections import deque
import math

import cv2
from gymnasium import spaces
import mujoco
import numpy as np

from envs import arena
from envs import motor
from envs.tasks.line_follower_env import CAM_RES
from envs.tasks.line_follower_env import CONTROL_HZ
from envs.tasks.line_follower_env import FALL_HEIGHT
from envs.tasks.line_follower_env import LineFollowerEnv
from envs.tasks.line_follower_env import MAX_EPISODE_STEPS
from envs.tasks.line_follower_env import MAX_LINE_LOST_STEPS
from envs.tracks import circle_waypoints
from envs.tracks import hairpin_waypoints
from envs.tracks import oval_waypoints
from envs.tracks import s_curve_waypoints
from envs.tracks import wave_waypoints

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

# --- What the policy is actually given ------------------------------------------------------
# Two line readings rather than raw pixels, because this env exists to teach RL as a *control*
# problem on hardware that already solves the perception half. The laptop that drives the real
# rover receives a JPEG and runs exactly this extraction in OpenCV, so what the policy sees in
# sim is what it will see deployed -- no CNN to train and none to run at 10 Hz.
#
# Measured justification, from the goal-nav ablation: at equal step counts a vector-observation
# policy reached 54.8% task success while the same policy taking the raw image reached ~10%.
# The visual encoder was the entire difference. Handing over the centroid costs nothing that
# the physical rover would not have computed anyway.
#
# NEAR is the position error the controller closes on. FAR is curvature preview -- the thing
# every competitive line follower uses to know a bend is coming, and the input that makes
# slowing into corners possible instead of reacting after the fact.
# Both are 100% available when the rover is on the line. Under domain randomization the
# further band washes out first -- pixelation and brightness jitter erase a thin distant line
# -- so FAR sits only moderately ahead: measured over straight-line rollouts, (0.20, 0.40) is
# seen ~48% of the time against ~21% for (0.35, 0.55). A preview that is usually missing is
# not a preview.
NEAR_BAND = (0.00, 0.15)  # fractions of frame height, measured up from the bottom
FAR_BAND = (0.20, 0.40)

# Steps of line/encoder history handed to the policy. A memoryless controller cannot tell a
# line drifting left from one already left and coming back, and cannot estimate its own
# turn rate from a single encoder sample.
OBS_HISTORY = 4

# Reward for losing sight of the line entirely. Deliberately larger than a step of centring
# reward is worth: on the real rover, off the line means the run is over.
LINE_LOST_PENALTY = 2.0

# How much staying on the line is worth, per step, relative to the base class's
# PROGRESS_WEIGHT of 100 per lap. The inherited 0.3 was set when the centring signal did not
# work at all, so it was never really chosen: measured on a trained policy, 72.2% of its return
# came from progress and 27.8% from centring, i.e. it was paid roughly 4x more for going far
# than for going straight. It did exactly that -- 6.3 cm mean deviation from a 3 cm line
# against the classical follower's 3.2 cm.
CENTER_WEIGHT_REAL = 1.5

# Centring pays only while the rover is actually moving -- a gate, not a multiplier.
#
# The obvious exploit to block is a rover that parks on the line and farms a flat per-step
# centring reward; this env has no step penalty to discourage that. Scaling the term by speed
# blocks it, and was tried: it shifted the reward split from 28% centring to 67% exactly as
# intended, and made tracking *worse*, 6.3 cm to 7.4 cm mean deviation. Scaling pays for speed,
# and speed costs accuracy -- the policy saturated the speed reference 30% of the time and
# could no longer turn tightly enough to hold the line, where the classical follower holds a
# fixed 0.10 m/s and tracks at 3.2 cm.
#
# A gate keeps parking worthless without paying anything extra for going faster than this.
# Progress and COMPLETION_BONUS still reward finishing, so there is a reason to move on.
MIN_PROGRESS_SPEED = 0.04  # m/s, well under the classical follower's 0.10

# --- Action space: (linear, angular), from the Webots controller's design -------------------
# One dimension is speed and one is turning, both independently useful. Per-wheel commands make
# the policy first discover that the sum is speed and the difference is turn -- a rotated basis
# it has to learn before it can learn control at all.
#
# The limits are this rover's, not the Webots model's. Its controller allowed 0.35 m/s, which
# is 10.45 rad/s at our wheel radius: past both the actuator range and the real rover's ceiling,
# so the top of that action range was not physically reachable. Ours come from
# rover_control/rover.py: 0.25 m/s, and the angular limit that both wheels opposed at maximum
# can produce.
# Geometry from rover.xml, via the modules that already own it.
WHEEL_RADIUS = motor.WHEEL_RADIUS
WHEEL_SEPARATION = arena.WHEEL_BASE

MAX_LINEAR_VEL = motor.MAX_WHEEL_SPEED * WHEEL_RADIUS  # 0.25 m/s
MAX_ANGULAR_VEL = 2.0 * motor.MAX_WHEEL_SPEED * WHEEL_RADIUS / WHEEL_SEPARATION  # ~3.08 rad/s

# Line orientation, fitted to the largest contour. The other half of the Webots design: the
# policy learns which way the line *runs*, not just where it is, so it can anticipate a bend.
# Without it the only curvature cue is the near/far centroid difference, which is implicit and
# weak -- consistent with the previous policy cutting corners at 0.159 m/s while the classical
# follower holds 0.101 and tracks twice as tightly.
#
# Thresholding stays relative to frame brightness rather than the Webots controller's absolute
# 100: that is what keeps the line visible through the shade randomization, and the absolute
# version is what made the faithful port lose the oval from 7 of 11 on-track poses.
ANGLE_CONTOUR_MIN_AREA = 3.0  # px^2; below this the fit is noise


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

# Held-out: never trained on, only measured. Two training tracks cannot tell you whether a
# policy follows a line or has fit an ellipse and a sine, and "no evaluation failures on
# varying tracks" is the bar before this goes on hardware -- so these are the varying tracks.
# Constant curvature, three times the training bend frequency, and one sustained bend tighter
# than anything in training. See envs/tracks.py for how each was sized.
EVAL_TRACKS_REAL = {
    "rover_line_circle_real.xml": circle_waypoints,
    "rover_line_wave_real.xml": wave_waypoints,
    "rover_line_hairpin_real.xml": hairpin_waypoints,
}


class LineFollowerRealEnv(LineFollowerEnv):
    """LineFollowerEnv, constrained to what the real rover can actually sense and command,
        plus domain randomization over the physical properties sim can't otherwise get right.

    This is the deployable variant: everything it senses, the physical rover senses, and
        everything it commands, the physical rover can execute.

        Observation, all of it computable on the laptop that drives the real rover:
          line      -- the last OBS_HISTORY readings of (near error, near angle, near seen,
                       far error, far seen). Near is the position error and the fitted
                       orientation of the track; far is curvature preview. Extracted from the
                       camera by the same centroid the classical follower uses, so no CNN is
                       trained here and none has to run at 10 Hz on deployment.
          encoders  -- the last OBS_HISTORY left/right wheel tick deltas, quantized and noisy,
                       matching the raw quadrature counts the firmware's "e" command returns.

        No image, and no ground-truth position. LineFollower-v0 keeps the raw-pixel observation
        for anyone who wants to study learning perception end to end; this one is about control.

        Action: normalized [-1, 1] mapping to (linear, angular) velocity, mixed to wheel speeds
        inside the env and passed through the motors' real envelope -- top speed and a dead zone
        below which the wheels do not turn at all. Normalized rather than physical units because
        SB3's Gaussian policy starts at std ~= 1 around zero, so m/s and rad/s put almost every
        sampled command outside the useful range.

        RESULTS, all at 300k steps with the same reward, so the comparison isolates design:

            per-wheel actions, no line angle : 10.0% completion, 6.1 cm mean deviation
            (v, omega) actions + line angle  : 86.7% completion, 5.6 cm on completed episodes
            classical PID, for reference     : 3.2 cm s-curve / 3.7 cm oval

        The design change bought *completion*, not precision, which matches what the two
        additions provide: (v, omega) makes the control problem separable -- one dimension is
        speed, one is turning -- and the line angle says which way the track runs so a bend can
        be anticipated. Neither makes the steering finer. Reward per step reached 1.52 against
        the classical follower's 1.54.

        The completions are real rather than a reward hack: sampled directly, completed episodes
        have the line in view 90.3% of the time, so the rover follows the track rather than
        circling it and letting the arc-length projection register laps.

        The open problem is variance. Aggregated over 30 evaluations the mean is 11.6 cm with a
        worst case of 118.3 cm, well outside a 1.2 x 0.8 m track, because roughly one episode in
        seven fails badly and drags the average. That is a tail-risk question to diagnose on its
        own terms, not something more reward tuning of the common case will reach.

        Wheel torque comes from envs/motor.py's JGA25-371 DC-motor model (BAM friction + a
        hand-derived electrical model from the datasheet) instead of MuJoCo's built-in velocity
        servo, so this env's scenes (assets/robots/rover/*_real.xml) use raw torque actuators.
    """

    TRACKS = TRACKS_REAL
    EVAL_TRACKS = EVAL_TRACKS_REAL

    def __init__(self, render_mode=None, domain_randomize=True, track=None):
        super().__init__(render_mode=render_mode, domain_randomize=domain_randomize, track=track)

        self.observation_space = spaces.Dict({
            # (near error, near angle, near seen, far error, far seen), most recent first.
            "line": spaces.Box(low=-1.0, high=1.0, shape=(OBS_HISTORY, 5), dtype=np.float32),
            "encoders": spaces.Box(low=-1.0, high=1.0, shape=(OBS_HISTORY, 2), dtype=np.float32),
        })
        # Normalized; scaled to (linear, angular) in step(). Normalized rather than physical
        # units because SB3's Gaussian policy starts at std ~= 1 around zero -- an action space
        # in m/s and rad/s puts almost every sampled command outside the useful range.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._history = deque(maxlen=OBS_HISTORY)

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
            # Unit-to-unit motor variation, so the policy cannot assume one exact envelope.
            self._max_omega = motor.MAX_WHEEL_SPEED * rng.uniform(0.85, 1.15)
            self._min_omega = motor.MIN_WHEEL_SPEED * rng.uniform(0.70, 1.30)
            wheel_friction = rng.uniform(*WHEEL_FRICTION_RANGE)
            self.model.cam_fovy[self._cam_id] = rng.uniform(*FOVY_RANGE)
            self._brightness = rng.uniform(*BRIGHTNESS_RANGE)
            self._white_balance = rng.uniform(*WHITE_BALANCE_RANGE, size=3)
            self._pixel_noise_std = rng.uniform(*PIXEL_NOISE_STD_RANGE)
            self._resolution_level = int(rng.choice(RESOLUTION_LEVELS))
        else:
            self._max_omega = motor.MAX_WHEEL_SPEED
            self._min_omega = motor.MIN_WHEEL_SPEED
            wheel_friction = 1.0
            self.model.cam_fovy[self._cam_id] = 60.0  # rover.xml's own top_cam default
            self._brightness = 1.0
            self._white_balance = np.ones(3)
            self._pixel_noise_std = 0.0
            self._resolution_level = CAM_RES  # full resolution, no pixelation
        for gid in self._wheel_geom_ids:
            self.model.geom_friction[gid, 0] = wheel_friction

        self._history.clear()
        self._cached_image = self._capture_processed_image()

        return self._get_real_obs(), info

    def step(self, action):
        # [-1, 1] in, (linear m/s, angular rad/s) out, then differential-drive mixing to wheel
        # speeds and the motors' real envelope. Linear maps to [0, MAX] rather than
        # [-MAX, MAX]: a zero-mean policy then starts at half speed forward, which is a useful
        # prior and keeps it out of the dead zone while it explores.
        action = np.clip(action, -1.0, 1.0)
        linear = MAX_LINEAR_VEL * (action[0] + 1.0) / 2.0
        angular = MAX_ANGULAR_VEL * action[1]
        left = (linear - angular * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
        right = (linear + angular * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
        # rover.xml's left axle reads negative when its wheel rolls the rover forward.
        action = np.array([-left, right])
        noisy_action = action + self.np_random.normal(0.0, self._action_noise_std, size=2)
        noisy_action = motor.apply_actuator_envelope(noisy_action, self._max_omega, self._min_omega)

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
            drive, electrical_damping = motor.motor_drive_and_damping(target_omega, measured_omega)
            self.data.ctrl[:] = drive
            for i, dof_adr in enumerate(self._wheel_dof_adr):
                motor.apply_friction(
                    self._friction_models[i],
                    self.model,
                    dof_adr,
                    measured_omega[i],
                    extra_damping=float(electrical_damping[i]),
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
            reward = -LINE_LOST_PENALTY
        else:
            self._lost_steps = 0
            # Centred *and* moving, but with no extra paid for extra speed. See
            # MIN_PROGRESS_SPEED above for why this is a gate rather than a multiplier.
            moving = abs(self._forward_speed()) >= MIN_PROGRESS_SPEED
            reward = CENTER_WEIGHT_REAL * (1.0 - abs(error)) if moving else 0.0
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
        self.renderer.update_scene(self.data, camera="top_cam")
        rgb = self.renderer.render().astype(np.float32) * self._white_balance
        gray = rgb.mean(axis=-1, keepdims=True) * self._brightness
        gray += self.np_random.normal(0.0, self._pixel_noise_std, size=gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return _pixelate(gray, self._resolution_level, CAM_RES)

    def _forward_speed(self):
        """Forward speed in m/s from the base velocimeter -- the same sensor the viewer plots
        and the same quantity a real rover would publish. Forward is the rover's local -Y."""
        adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "base_velocity")
        ]
        return -float(self.data.sensordata[adr + 1])

    def _band_error(self, band, with_angle=False):
        """(error, seen) -- or (error, angle, seen) -- for one horizontal slice of the frame,
        measured up from the bottom.

        The same extraction the deployed laptop will run on a JPEG: threshold relative to frame
        brightness so it survives the shade randomization, then a centroid weighted by how many
        dark pixels each column holds. With `with_angle`, additionally fit a line to the largest
        contour so the policy learns the track's orientation, not only its offset."""
        gray = self._cached_image[:, :, 0].astype(np.float32)
        lo = int(CAM_RES * (1.0 - band[1]))
        hi = int(CAM_RES * (1.0 - band[0]))
        window = gray[lo:hi, :]
        dark = window < (gray.mean() - 25.0)
        weights = dark.sum(axis=0).astype(np.float64)
        if weights.sum() == 0:
            return (0.0, 0.0, 0.0) if with_angle else (0.0, 0.0)
        centroid = float((np.arange(CAM_RES) * weights).sum() / weights.sum())
        error = (centroid - CAM_RES / 2) / (CAM_RES / 2)
        if not with_angle:
            return error, 1.0

        angle = 0.0
        contours, _ = cv2.findContours(
            dark.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= ANGLE_CONTOUR_MIN_AREA and len(largest) >= 2:
                vx, vy, _, _ = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                angle = float(np.arctan2(vx, vy)) / math.pi
        return error, float(np.clip(angle, -1.0, 1.0)), 1.0

    def _line_error(self, gray_obs, band=None):
        """Near-band centring error, or None when the line is not in view.

        Overrides the base class so the reward is scored on the same near-field reading the
        policy is given, rather than on a whole-frame centroid."""
        near_error, seen = self._band_error(NEAR_BAND)
        return near_error if seen else None

    def _get_real_obs(self):
        """Line features and encoder ticks, both as short histories.

        The encoder read models the real sensor rather than an idealized velocity: the
        firmware's "e" command returns raw cumulative quadrature counts, so a client derives
        motion by polling twice and differencing. Angle-domain noise is applied before
        quantizing to whole ticks, so the value carries the same discreteness a real read
        would."""
        wheel_angle = self.data.qpos[self._wheel_qpos_adr]
        delta_angle = wheel_angle - self._prev_wheel_angle
        self._prev_wheel_angle = wheel_angle.copy()

        noise_rad = self.np_random.normal(
            0.0, self._encoder_noise_std * self._physics_dt * self._decimation, size=2
        )
        ticks = np.round((delta_angle + noise_rad) * ENCODER_CPR_WHEEL / (2 * np.pi))

        near_error, near_angle, near_seen = self._band_error(NEAR_BAND, with_angle=True)
        far_error, far_seen = self._band_error(FAR_BAND)
        self._history.appendleft((
            np.array([near_error, near_angle, near_seen, far_error, far_seen], dtype=np.float32),
            (ticks / ENCODER_TICKS_BOUND).astype(np.float32),
        ))
        frames = list(self._history)
        frames += [frames[-1]] * (OBS_HISTORY - len(frames))

        return {
            "line": np.clip(np.stack([f[0] for f in frames]), -1.0, 1.0),
            "encoders": np.clip(np.stack([f[1] for f in frames]), -1.0, 1.0),
        }
