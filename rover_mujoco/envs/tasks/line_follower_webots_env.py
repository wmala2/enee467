"""A faithful port of the Webots line-following RL environment, with no domain randomization.

Copied from webots_ws/controllers/line_follower_camera_rl/diff_drive_robot.py, which learned
this task in roughly 100k steps and generalized across tracks. This env exists to isolate
*design* from everything else: if it learns quickly here too, the gap in LineFollowerReal-v0 is
its observation and action design rather than its training budget or its randomization.

Three things it does differently from LineFollowerReal-v0, in the order they probably matter:

1. The action is (linear velocity, angular velocity), not per-wheel speeds. One dimension is
   speed and one is turning, both independently useful. Per-wheel forces the policy to first
   discover that the sum is speed and the difference is turn -- a rotated basis it has to learn
   before it can learn control. Still deployable: the differential-drive mixing to wheel speeds
   is two lines, and rover_control/rover.py takes left/right at the boundary anyway.

2. The observation carries the line's *angle*, not just its lateral offset. Fitted to the
   largest contour with cv2.fitLine, so the policy knows which way the line runs and can
   anticipate a bend rather than react to it. LineFollowerReal-v0 only ever knew where the line
   was, which is why it cuts corners.

3. Velocities are given directly instead of raw encoder ticks. Both are honest to the hardware
   -- the real rover derives velocity from ticks -- but one is far easier to learn from.

Deliberately absent: any domain randomization. This is the clean-room comparison.

RESULT: it does not learn here. 100k steps took the return from -10787 to -7966, still deeply
negative, where the original solved the task in about 25k. Two measured reasons, neither of
which is the observation or action design this port exists to test:

1. Under their reward, in our sim, *standing still is optimal*. Scored directly: a stopped
   rover earns +2.87/step, a PID on (v, omega) earns -0.86, driving straight earns -4.31.
   Parked on the line collects 2*alignment + 1*heading = 3.0/step risk-free, while driving at
   0.10 m/s adds only 5 * (0.10 / 0.35) = 1.43 against a -10/step line-loss risk. PPO found
   the same degenerate optimum. Their weights are only safe when losing the line is rare --
   the -10 is meant to be hypothetical, not routine.

2. Here it is routine, because of the oval. With the rover placed *on* the track, the line is
   visible from 11/11 sampled poses on the s-curve but only 4/11 on the oval, which is
   1.2 x 0.8 m and tight enough that at this camera's 60 degree tilt the line curves out of
   frame. Their ROI is the bottom 30% with an absolute threshold of 100; LineFollowerReal-v0
   uses the bottom 15% with a threshold relative to frame brightness, and keeps the line 100%
   of the time on both tracks with the same camera.

So the Webots design is not magic and ours is not uniquely broken: their reward assumes
reliable perception, and our tracks are tighter relative to the camera's usable range than
theirs were. The (v, omega) action space and the line-angle observation still look like real
improvements, but they could never show through a reward whose optimum is to stop.
"""

import math
import os
import random
from typing import ClassVar

import cv2
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from envs.tasks.line_follower_env import CONTROL_HZ
from envs.tasks.line_follower_env import FALL_HEIGHT
from envs.tracks import oval_waypoints
from envs.tracks import s_curve_waypoints

# --- The rover's limits, as the Webots controller declared them ----------------------------
MIN_LINEAR_VEL = 0.0  # m/s
MAX_LINEAR_VEL = 0.35  # m/s
MIN_ANGULAR_VEL = -2.54  # rad/s
MAX_ANGULAR_VEL = 2.54  # rad/s

# This rover's own geometry, from rover.xml, rather than the Webots model's slightly different
# numbers -- the point is to test their design on our robot.
WHEEL_RADIUS = 0.0335
WHEEL_SEPARATION = 0.1626

# --- Image processing, copied from their process_image() -----------------------------------
CAM_RES = 64
ROI_RATIO = 0.7  # use the bottom 30% of the frame
BINARY_THRESHOLD = 100  # cv2.THRESH_BINARY_INV cutoff

# --- Reward weights, copied verbatim -------------------------------------------------------
ALIGNMENT_WEIGHT = 2.0
HEADING_WEIGHT = 1.0
MOTION_WEIGHT = 5.0
SMOOTHNESS_WEIGHT = -0.1
LINE_LOST_PENALTY = 10.0

# Wheel acceleration limit, in rad/s^2, taken from scripts/teleop_rover.py's MAX_ACCEL.
#
# Not a tuning knob and not a deviation from the Webots design: this rover physically cannot
# change wheel speed instantly, and teleop_rover.py already documents why the ramp exists --
# "a direction key or a '+' press would snap straight to the target speed, which is exactly
# what caused the wheelie popups". A MuJoCo <velocity> servo applies whatever torque it takes
# to hit its target on the next step, so without the ramp the policy can flip the rover simply
# by changing its mind. Measured: random actions tipped it in 12 of 12 episodes, ending them
# after ~76 steps, which is a plant difference from Webots' motors rather than anything about
# the observation or reward being compared here.
MAX_WHEEL_ACCEL = 15.0

# Their steps_per_episode was 20000, which at 10 Hz is a 33-minute episode and only about five
# resets across a 100k-step run. Shortened so the policy sees both tracks often; this is the
# one place the port deliberately diverges, and it should make learning easier, not harder.
MAX_EPISODE_STEPS = 2000

TRACKS = {
    "rover_line_oval.xml": oval_waypoints,
    "rover_line_s_curve.xml": s_curve_waypoints,
}


def normalize_to_range(value, in_lo, in_hi, out_lo, out_hi, clip=True):
    """Their utilities.normalize_to_range, reproduced so the observation scaling matches."""
    scaled = (value - in_lo) / (in_hi - in_lo) * (out_hi - out_lo) + out_lo
    return float(np.clip(scaled, out_lo, out_hi)) if clip else float(scaled)


def velocity_to_wheels(linear, angular):
    """(v, omega) -> left/right wheel angular speeds in rad/s, standard differential drive."""
    left = (linear - angular * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    right = (linear + angular * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    return left, right


class LineFollowerWebotsEnv(gym.Env):
    """The Webots line follower's design, on this repo's rover and tracks, without DR.

    Observation (4,), each normalized:
        centroid_x_error  [-1, 1]  lateral offset of the line's centroid in the ROI
        line_angle        [-1, 1]  orientation of the fitted line
        linear_velocity   [ 0, 1]
        angular_velocity  [-1, 1]

    Action (2,): linear velocity m/s, angular velocity rad/s.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, track=None):
        super().__init__()
        self.render_mode = render_mode
        self._fixed_track = track

        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([MIN_LINEAR_VEL, MIN_ANGULAR_VEL], dtype=np.float32),
            high=np.array([MAX_LINEAR_VEL, MAX_ANGULAR_VEL], dtype=np.float32),
            dtype=np.float32,
        )

        self._scene = None
        self._load_scene(track or next(iter(TRACKS)))
        self._episode_steps = 0
        self._wheel_targets = np.zeros(2)

    def _load_scene(self, scene_file):
        if self._scene == scene_file:
            return
        path = os.path.join(os.path.dirname(__file__), "../../assets/robots/rover", scene_file)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.model, height=CAM_RES, width=CAM_RES)
        self._physics_dt = self.model.opt.timestep
        self._decimation = max(1, round(1.0 / (CONTROL_HZ * self._physics_dt)))
        self._waypoints_fn = TRACKS[scene_file]
        self._scene = scene_file

    def close(self):
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None
        super().close()

    def process_image(self, img):
        """Their process_image: threshold the bottom ROI, take the largest contour, fit a line
        to it for the angle and use image moments for the centroid.

        Contours rather than a column-weighted centroid because the largest connected blob
        rejects speckle that an average over columns would be dragged by."""
        roi = img[int(CAM_RES * ROI_RATIO) :, :, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0, False

        contour = max(contours, key=cv2.contourArea)
        # fitLine returns a (4, 1) array; ravel before unpacking.
        vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        line_angle = float(np.arctan2(vx, vy))

        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"] if moments["m00"] > 0 else CAM_RES / 2
        return float(cx - CAM_RES / 2), line_angle, True

    def _velocities(self):
        """Body linear (forward) and angular (yaw) velocity. Forward is the rover's local -Y."""
        linear = -float(self.data.qvel[1])
        angular = float(self.data.qvel[5])
        return linear, angular

    def _get_obs(self):
        self.renderer.update_scene(self.data, camera="top_cam")
        centroid, angle, seen = self.process_image(self.renderer.render())
        self._seen = seen
        self._centroid = normalize_to_range(centroid, -CAM_RES / 2, CAM_RES / 2, -1.0, 1.0)
        self._angle = normalize_to_range(angle, -math.pi, math.pi, -1.0, 1.0)
        linear, angular = self._velocities()
        self._linear = normalize_to_range(linear, MIN_LINEAR_VEL, MAX_LINEAR_VEL, 0.0, 1.0)
        self._angular = normalize_to_range(angular, MIN_ANGULAR_VEL, MAX_ANGULAR_VEL, -1.0, 1.0)
        return np.array(
            [self._centroid, self._angle, self._linear, self._angular], dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_scene(self._fixed_track or random.choice(list(TRACKS)))
        mujoco.mj_resetData(self.model, self.data)

        (x0, y0), (x1, y1) = self._waypoints_fn()[:2]
        tx, ty = x1 - x0, y1 - y0
        norm = math.hypot(tx, ty)
        yaw = math.atan2(tx / norm, -ty / norm)
        self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] = x0, y0, 0.1
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

        self._episode_steps = 0
        self._wheel_targets = np.zeros(2)
        return self._get_obs(), {}

    def step(self, action):
        linear = float(np.clip(action[0], MIN_LINEAR_VEL, MAX_LINEAR_VEL))
        angular = float(np.clip(action[1], MIN_ANGULAR_VEL, MAX_ANGULAR_VEL))
        left, right = velocity_to_wheels(linear, angular)
        # rover.xml's left axle reads negative when its wheel rolls the rover forward; see
        # envs/arena.py's integrate_odometry for the same convention undone in reverse.
        commanded = np.array([-left, right])
        # Ramp toward the command rather than snapping to it -- see MAX_WHEEL_ACCEL.
        max_delta = MAX_WHEEL_ACCEL * self._physics_dt
        for _ in range(self._decimation):
            step = np.clip(commanded - self._wheel_targets, -max_delta, max_delta)
            self._wheel_targets += step
            self.data.ctrl[:] = self._wheel_targets
            mujoco.mj_step(self.model, self.data)
        self._episode_steps += 1

        obs = self._get_obs()

        # Their get_reward, verbatim in structure and weights.
        alignment = 1.0 - abs(self._centroid)
        heading = math.cos(self._angle * math.pi)
        motion = max(0.0, self._linear)
        smoothness = abs(self._angular)
        reward = (
            ALIGNMENT_WEIGHT * alignment
            + HEADING_WEIGHT * heading
            + MOTION_WEIGHT * motion
            + SMOOTHNESS_WEIGHT * smoothness
        )
        # Their check was `centroid_x_error == 0`, which also fires for a perfectly centred
        # line. Using the contour-found flag instead: same intent, without the false positive.
        if not self._seen:
            reward -= LINE_LOST_PENALTY

        terminated = bool(self.data.qpos[2] < FALL_HEIGHT)
        truncated = self._episode_steps >= MAX_EPISODE_STEPS
        return obs, reward, terminated, truncated, {"line_seen": self._seen}
