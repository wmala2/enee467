import math
import os
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from envs.tracks import oval_waypoints, s_curve_waypoints

# Onboard camera image observations, not ground-truth position: this is what makes the
# task realistic for sim-to-real transfer, at the cost of much slower CPU rollouts than
# RoverEnv's vector observations — keep the resolution small.
CAM_RES = 64
LINEAR_SPEED = 3.0
MAX_LINE_LOST_STEPS = 20
MAX_EPISODE_STEPS = 500
FALL_HEIGHT = 0.03  # base_link z below this means the rover tipped over

TRACKS = {
    "rover_line_oval.xml": oval_waypoints,
    "rover_line_s_curve.xml": s_curve_waypoints,
}


class LineFollowerEnv(gym.Env):
    """Camera-only line follower. Reward and termination are both computed from the
    rendered onboard image, the same signal a real deployed rover would have to use —
    no privileged (x, y) ground truth leaks into the policy or its reward."""

    metadata = {"render_modes": ["human"], "render_fps": 50}
    TRACKS = TRACKS  # class attribute so subclasses (e.g. LineFollowerRealEnv) can point
    # at different scene files (different actuators) without re-implementing __init__

    def __init__(self, render_mode=None, domain_randomize=True):
        super().__init__()
        self.domain_randomize = domain_randomize
        self.render_mode = render_mode

        scene_file, self._waypoints_fn = random.choice(list(self.TRACKS.items()))
        model_path = os.path.join(
            os.path.dirname(__file__), "../../assets/robots/rover", scene_file
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=CAM_RES, width=CAM_RES)

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(CAM_RES, CAM_RES, 1), dtype=np.uint8
        )

        # Baseline appearance to randomize around, and the ids of the geoms/light/camera
        # domain randomization is allowed to touch each reset
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._line_geom_ids = [i for i in range(self.model.ngeom) if self.model.geom_group[i] == 3]
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
        self._default_cam_pos = self.model.cam_pos[self._cam_id].copy()
        self._default_cam_quat = self.model.cam_quat[self._cam_id].copy()
        self._default_light_diffuse = self.model.light_diffuse.copy()

        self._lost_steps = 0
        self._episode_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.domain_randomize:
            self._randomize_appearance()

        (x0, y0), quat = self._start_pose()
        self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] = x0, y0, 0.1
        self.data.qpos[3:7] = quat
        mujoco.mj_forward(self.model, self.data)

        self._lost_steps = 0
        self._episode_steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._episode_steps += 1

        obs = self._get_obs()
        error = self._line_error(obs)

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

    def _start_pose(self):
        """Spawn at the track's first waypoint, yawed to face the second one (matches
        scripts/line_follower.py's convention so the PID and RL agents start identically)."""
        (x0, y0), (x1, y1) = self._waypoints_fn()[:2]
        tx, ty = x1 - x0, y1 - y0
        norm = math.hypot(tx, ty)
        tx, ty = tx / norm, ty / norm
        yaw = math.atan2(-tx, ty)
        quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        return (x0, y0), quat

    def _randomize_appearance(self):
        """Jitter floor/line color, light intensity, and camera mount pose around their
        defaults. This is the domain randomization step: train on many plausible-looking
        variants of the scene so the policy can't overfit to one exact rendering, which is
        the point that matters for a policy meant to eventually run on a real camera."""
        rng = self.np_random

        # Floor: any light, roughly-neutral surface color
        floor_shade = rng.uniform(0.5, 0.95)
        self.model.geom_rgba[self._floor_geom_id, :3] = floor_shade
        self.model.geom_rgba[self._floor_geom_id, 3] = 1.0

        # Line: always dark, but not a fixed dark value (marker wear, print contrast, etc.)
        line_shade = rng.uniform(0.0, 0.2)
        for gid in self._line_geom_ids:
            self.model.geom_rgba[gid, :3] = line_shade
            self.model.geom_rgba[gid, 3] = 1.0

        # Lighting intensity
        self.model.light_diffuse[:] = self._default_light_diffuse * rng.uniform(0.6, 1.3)

        # Camera mount tolerance: a few mm of position slop, a few degrees of tilt slop
        pos_jitter = rng.uniform(-0.005, 0.005, size=3)
        self.model.cam_pos[self._cam_id] = self._default_cam_pos + pos_jitter
        angle_jitter = math.radians(rng.uniform(-3, 3))
        axis = np.array([1.0, 0.0, 0.0])
        jitter_quat = np.array([
            math.cos(angle_jitter / 2),
            *(axis * math.sin(angle_jitter / 2)),
        ])
        mujoco.mju_mulQuat(self.model.cam_quat[self._cam_id], self._default_cam_quat, jitter_quat)

    def _get_obs(self):
        self.renderer.update_scene(self.data, camera="top_cam")
        rgb = self.renderer.render()
        gray = rgb.mean(axis=-1, keepdims=True).astype(np.uint8)
        return gray

    def _line_error(self, gray_obs):
        """Normalized horizontal offset of the line's centroid from image center, in
        [-1, 1], or None if no line pixels are visible. Uses a threshold relative to the
        frame's own mean brightness (not a fixed value) so it stays correct across the
        randomized floor/line shades above — the same reason a real line-follower would
        use adaptive thresholding rather than a hardcoded pixel value."""
        gray = gray_obs[:, :, 0].astype(np.float32)
        dark_mask = gray < (gray.mean() - 25.0)
        cols = np.where(dark_mask.any(axis=0))[0]
        if len(cols) == 0:
            return None
        centroid = cols.mean()
        return (centroid - CAM_RES / 2) / (CAM_RES / 2)
