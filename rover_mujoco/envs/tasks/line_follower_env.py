import math
import os
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from envs import tracks
from envs.tracks import oval_waypoints, s_curve_waypoints

# Onboard camera image observations, not ground-truth position: this is what makes the
# task realistic for sim-to-real transfer, at the cost of much slower CPU rollouts than
# RoverEnv's vector observations — keep the resolution small.
CAM_RES = 64
LINEAR_SPEED = 3.0
MAX_LINE_LOST_STEPS = 20

# Control-loop rate: a fresh action decision every CONTROL_HZ, decimated down from the
# physics integrator's much finer 500 Hz (0.002s) timestep — no real actuator can be
# commanded that fast, and (found via an actual training run) letting PPO issue a brand
# new target wheel velocity every single physics tick let it discover a degenerate
# "vibrate in place" policy: whipsawing the command every 2ms kept the line centered
# (near-zero net displacement, so near-zero image error) far more easily than actually
# learning to drive, and since the reward for that was already ~97% of max from the very
# first rollout, training never had pressure to move past it. 10 Hz matches the real
# rover's camera/command-rate constraint (see LineFollowerRealEnv).
CONTROL_HZ = 10.0

# One episode = enough control decisions to actually traverse a track (previously 500
# *physics* steps at 0.002s was only 1 simulated second total — nowhere near enough time
# to drive either track, which is the other half of why standing still used to look
# reward-optimal). 300 control steps @ CONTROL_HZ = 30 simulated seconds.
MAX_EPISODE_STEPS = 300
FALL_HEIGHT = 0.03  # base_link z below this means the rover tipped over

# Reward weights. Finishing the track (as many times as possible) is the actual goal —
# speed/smoothness are secondary — so PROGRESS_WEIGHT (forward progress along the track's
# waypoints, ground-truth position) is the dominant term and CENTER_WEIGHT (line-centering,
# the only signal a real deployed policy would have) is a smaller shaping term that keeps
# the rover from cutting off the line while chasing progress. Reward-only privilege is fine
# here: reward doesn't exist at deployment, only the policy's (camera+encoder) observation
# does, and that stays unprivileged as before.
PROGRESS_WEIGHT = 100.0   # per step: PROGRESS_WEIGHT * (forward arc-length delta / track length)
                          # normalized so one full lap/traversal always sums to ~PROGRESS_WEIGHT,
                          # regardless of the oval vs. s-curve's different physical lengths
CENTER_WEIGHT = 0.3       # per step: CENTER_WEIGHT * (1 - abs(line-centering error))
COMPLETION_BONUS = 50.0   # flat bonus each time a lap (closed track) or the end (open track) is reached

TRACKS = {
    "rover_line_oval.xml": oval_waypoints,
    "rover_line_s_curve.xml": s_curve_waypoints,
}


class LineFollowerEnv(gym.Env):
    """Camera-only line follower. The policy's observation is never privileged (no
    ground-truth position, only what a real deployed rover could sense). Reward *does*
    use ground-truth (x, y) position to score forward progress along the track — fine
    since reward is a training-only construct that doesn't exist at deployment, unlike
    the observation, which has to work with only what the real rover can sense."""

    metadata = {"render_modes": ["human"], "render_fps": 50}
    TRACKS = TRACKS  # class attribute so subclasses (e.g. LineFollowerRealEnv) can point
    # at different scene files (different actuators) without re-implementing __init__

    def __init__(self, render_mode=None, domain_randomize=True):
        super().__init__()
        self.domain_randomize = domain_randomize
        self.render_mode = render_mode

        scene_file, self._waypoints_fn = random.choice(list(self.TRACKS.items()))
        # Track geometry (and thus its arc-length table) is fixed for this env instance's
        # whole lifetime, same as scene_file/waypoints_fn above — only the per-episode
        # progress *state* (self._prev_arc_s etc., set in reset()) changes between resets.
        self._path_waypoints = self._waypoints_fn()
        self._path_cumlen, self._path_closed = tracks.path_length_table(self._path_waypoints)
        self._path_total_len = self._path_cumlen[-1]
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

        # Physics steps per control decision — see CONTROL_HZ above.
        self._physics_dt = self.model.opt.timestep
        self._decimation = max(1, round(1.0 / (CONTROL_HZ * self._physics_dt)))

    def close(self):
        """Release the offscreen renderer's GL context. Not just cleanup hygiene: MuJoCo's
        Renderer leaves that context in a broken state for whatever Renderer is constructed
        next *in the same process* if this is skipped — found by hand (every env after the
        first one in a process rendered solid black) tracing what looked like a spawn/DR bug
        but was actually this. Harmless for a single long-lived training worker (it only ever
        makes one renderer), but required for any script that constructs more than one
        LineFollowerEnv in one process (e.g. an eval loop over several episodes/tracks)."""
        self.renderer.close()
        super().close()

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

        # Track-progress reward state (see _track_progress_reward) — (x0, y0) is on the
        # path by construction (_start_pose spawns at its first waypoint), so this is ~0.
        # Only this initial fix searches the whole path (near_segment=None); every step
        # after stays windowed around wherever it last matched.
        self._prev_arc_s, self._prev_arc_segment = tracks.project_arc_length(
            self._path_waypoints, self._path_cumlen, (x0, y0)
        )
        self._unwrapped_progress = 0.0
        self._laps_completed = 0
        self._finished = False

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self._decimation):
            mujoco.mj_step(self.model, self.data)
        self._episode_steps += 1

        obs = self._get_obs()
        error = self._line_error(obs)
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
            self._lost_steps >= MAX_LINE_LOST_STEPS
            or self._episode_steps >= MAX_EPISODE_STEPS
        )
        return obs, reward, terminated, truncated, {}

    def _track_progress_reward(self):
        """Forward progress along the track's waypoints since the last step (ground-truth
        (x, y), projected onto the polyline — see envs/tracks.py), normalized by the
        track's own length so a full lap (closed track) or traversal (open track) always
        sums to ~PROGRESS_WEIGHT regardless of the oval vs. s-curve's different physical
        lengths, plus a flat COMPLETION_BONUS each time that actually happens.

        Returns (reward, finished): `finished` is True exactly once, the step an open
        (non-looping) track's far end is reached — the caller ends the episode there,
        successfully, rather than letting it idle past the end of the line."""
        s, self._prev_arc_segment = tracks.project_arc_length(
            self._path_waypoints, self._path_cumlen, self.data.qpos[:2],
            near_segment=self._prev_arc_segment, closed=self._path_closed,
        )
        delta = s - self._prev_arc_s
        if self._path_closed:
            # Shortest signed distance around the loop, so wrapping from ~total_len back
            # to ~0 registers as further forward progress, not a huge backward jump.
            total = self._path_total_len
            delta = (delta + total / 2) % total - total / 2
        self._prev_arc_s = s
        self._unwrapped_progress += delta

        reward = PROGRESS_WEIGHT * delta / self._path_total_len
        finished = False
        if self._path_closed:
            laps = int(self._unwrapped_progress // self._path_total_len)
            if laps > self._laps_completed:
                self._laps_completed = laps
                reward += COMPLETION_BONUS
        elif not self._finished and self._unwrapped_progress >= self._path_total_len:
            self._finished = True
            finished = True
            reward += COMPLETION_BONUS
        return reward, finished

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
