"""PPO line following with the same geometry and evaluation contract as the PID baseline."""

import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from envs.contract import observation_from_sensors
from envs.contract import wheel_targets
from envs.line_dynamics import LineDynamics
from envs.line_scene import build_model
from envs.line_scene import CAM_RES
from envs.line_scene import CONTROL_HZ
from envs.tracks import circle_waypoints
from envs.tracks import figure8_waypoints
from envs.tracks import goomba_waypoints
from envs.tracks import oval_waypoints
from envs.tracks import path_length_table
from envs.tracks import project_arc_length

TRACKS = {
    "circle": circle_waypoints,
    "figure8": figure8_waypoints,
    "goomba": goomba_waypoints,
    "oval": oval_waypoints,
}

# Share of the centering term taken from chassis deviation under reward_centering="both".
# The camera sits 0.108 m ahead of the body origin, so a policy that centers the camera puts
# the chassis inside the curve by roughly d^2 / 2R: about 3.8 cm at the figure-eight's 0.155 m
# minimum radius, against a 6 cm budget. Camera centering alone therefore corner-cuts, and
# measured peak deviations land on the lobe apexes rather than anywhere else on the lap.
# Chassis deviation alone removes the bias but loses the crossing, where both branches are in
# frame and only camera centering says which one this episode is on. A small chassis share
# keeps the branch lock and pays down the geometric offset.
CHASSIS_SHARE = 0.2
WHEEL_RADIUS = 0.03435
CRUISE_RAD_S = 3.0
MAX_STEERING_RAD_S = 4.0
MAX_WHEEL_RAD_S = 10.0


class LineFollowerPPOEnv(gym.Env):
    """Observe three camera centroids and wheel encoders; command forward speed and steering."""

    # Gymnasium reads render metadata from the environment class.
    metadata: dict = {"render_modes": ["rgb_array", "human"], "render_fps": 10}  # noqa: RUF012

    def __init__(
        self,
        track="circle",
        random_start=False,
        render_mode=None,
        max_steps=1200,
        reward_centering="camera",
        dynamics="nominal",
        dr_ranges=None,
    ):
        super().__init__()
        if track not in TRACKS:
            raise ValueError(f"Unknown track {track!r}; choose from {list(TRACKS)}")
        self.track_name = track
        self.random_start = random_start
        self.render_mode = render_mode
        self.max_steps = max_steps
        if reward_centering not in ("camera", "chassis", "both"):
            raise ValueError("reward_centering must be 'camera', 'chassis' or 'both'")
        self.reward_centering = reward_centering
        self.points = np.asarray(TRACKS[track]())
        self.lengths, _ = path_length_table(self.points)
        if dynamics not in ("nominal", "bam", "dr"):
            raise ValueError("dynamics must be 'nominal', 'bam', or 'dr'")
        if dr_ranges is not None and dynamics != "dr":
            raise ValueError("dr_ranges requires dynamics='dr'")
        self.dynamics_mode = dynamics
        self.model = build_model(self.points, 45, bam=dynamics != "nominal")
        self.dynamics = (
            LineDynamics(self.model, dynamics, dr_ranges) if dynamics != "nominal" else None
        )
        # MuJoCo's compiled exports lack type stubs in this installation.
        self.data = mujoco.MjData(self.model)  # ty: ignore[unresolved-attribute]
        self.renderer = mujoco.Renderer(self.model, height=CAM_RES, width=CAM_RES)
        self.substeps = round(1 / CONTROL_HZ / self.model.opt.timestep)
        self.dt = self.substeps * self.model.opt.timestep
        self.axles = [self.model.joint(name).qposadr[0] for name in ("left_axle", "right_axle")]
        self.viewer = None
        # Nine camera features plus two signed wheel-speed estimates, all bounded by [-1, 1].
        self.observation_space = spaces.Box(-1, 1, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.frame = np.zeros((CAM_RES, CAM_RES, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)  # ty: ignore[unresolved-attribute]
        # Seed all physical and sensor perturbations with Gymnasium's episode generator.
        if self.dynamics is not None:
            self.dynamics.reset(self.np_random, self.data)
        segment = int(self.np_random.integers(len(self.points) - 1)) if self.random_start else 0
        tangent = self.points[segment + 1] - self.points[segment]
        tangent /= np.linalg.norm(tangent)
        lateral = self.np_random.uniform(-0.01, 0.01)
        yaw = math.atan2(tangent[0], -tangent[1]) + self.np_random.uniform(
            -math.radians(5), math.radians(5)
        )
        self.data.qpos[:2] = self.points[segment] + lateral * np.array([-tangent[1], tangent[0]])
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        # Settle before starting the episode, matching the classical-controller evaluation.
        if self.dynamics is None:
            mujoco.mj_step(self.model, self.data, nstep=round(1 / self.model.opt.timestep))  # ty: ignore[unresolved-attribute]
        else:
            self.dynamics.advance(
                self.data, np.zeros(2), round(1 / self.model.opt.timestep), settling=True
            )
        mujoco.mj_forward(self.model, self.data)  # ty: ignore[unresolved-attribute]
        self.previous_angles = self.data.qpos[self.axles].copy()
        self.wheel_speeds = np.zeros(2)
        self.previous_action = np.array([-1, 0], dtype=np.float32)
        self.arc, self.segment = project_arc_length(
            self.points, self.lengths, self.data.qpos[:2], segment, window=4, closed=True
        )
        self.progress = 0.0
        self.steps = 0
        self.lost_steps = 0
        self.total_lost = 0
        self.deviations = []
        self.episode_return = 0.0
        self.finished = False
        observation = self._get_observation()
        return observation, {
            "track": self.track_name,
            "dynamics": self.dynamics_mode,
            "domain_parameters": self.dynamics.parameters if self.dynamics else {},
        }

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset the environment after an episode ends")
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        # CAD axle signs: forward motion is negative left and positive right.
        targets = wheel_targets(action)
        if self.dynamics is None:
            self.data.ctrl[:] = targets * [-1, 1]
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)  # ty: ignore[unresolved-attribute]
        else:
            self.dynamics.advance(self.data, targets, self.substeps)
        mujoco.mj_forward(self.model, self.data)  # ty: ignore[unresolved-attribute]
        self.steps += 1
        angles = self.data.qpos[self.axles].copy()
        self.wheel_speeds = (angles - self.previous_angles) / self.dt * [-1, 1]
        self.previous_angles = angles
        observation = self._get_observation()
        delta, deviation = self._track_progress()
        self.deviations.append(deviation)
        visible = bool(observation[2])
        self.lost_steps = 0 if visible else self.lost_steps + 1
        self.total_lost += not visible
        # Forward progress earns reward only while the line is visible; stopping earns no alignment.
        progress_reward = float(np.clip(delta / (CRUISE_RAD_S * WHEEL_RADIUS * self.dt), -2, 1))
        # Reward chassis tracking while retaining camera centering for controlled ablations.
        # "camera" and "chassis" are unchanged so earlier runs stay reproducible; "both" blends
        # them, see CHASSIS_SHARE for why neither alone is right.
        camera_error = abs(float(observation[0]))
        chassis_error = min(deviation / 0.06, 1.0)
        if self.reward_centering == "camera":
            error = camera_error
        elif self.reward_centering == "chassis":
            error = chassis_error
        else:
            error = (1 - CHASSIS_SHARE) * camera_error + CHASSIS_SHARE * chassis_error
        alignment = 1 - error if visible else 0
        motion_reward = progress_reward * (0.25 + 0.75 * alignment) if visible else -1.0
        smoothness_penalty = 0.02 * float(np.sum((action - self.previous_action) ** 2))
        reward = motion_reward - smoothness_penalty - 0.01
        self.previous_action = action.copy()
        # Task outcomes are geometric and temporal, independent of the shaped reward.
        reason = "running"
        if self.data.qpos[2] < 0.03:
            reason = "tipped"
        elif deviation > 0.06:
            reason = "off_track"
        elif self.lost_steps * self.dt >= 0.5:
            reason = "line_lost"
        elif self.progress >= self.lengths[-1] - 0.03:
            reason = "completed"
        elif self.steps >= self.max_steps:
            reason = "timeout"
        terminated = reason in ("tipped", "off_track", "line_lost", "completed")
        truncated = reason == "timeout"
        if terminated:
            reward += 10 if reason == "completed" else -5
        self.finished = terminated or truncated
        self.episode_return += reward
        info = {
            "track": self.track_name,
            "is_success": reason == "completed",
            "reason": reason,
            "progress_fraction": float(self.progress / self.lengths[-1]),
            "deviation_cm": deviation * 100,
            "line_visible": visible,
            "reward_motion": motion_reward,
            "reward_smoothness": -smoothness_penalty,
        }
        if self.finished:
            info.update({
                "mean_deviation_cm": float(np.mean(self.deviations) * 100),
                "max_deviation_cm": float(np.max(self.deviations) * 100),
                "line_loss_frames": self.total_lost,
                "duration_s": self.steps * self.dt,
                "dynamics": self.dynamics_mode,
                "domain_parameters": self.dynamics.parameters if self.dynamics else {},
            })
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        # Camera pixels and finite differences of encoder angles are the only policy inputs.
        self.renderer.update_scene(self.data, camera="top_cam")
        self.frame = self.renderer.render().copy()
        speeds = self.wheel_speeds
        if self.dynamics is not None:
            self.frame, speeds = self.dynamics.sensors(self.frame, speeds)
        return observation_from_sensors(self.frame, speeds)

    def _track_progress(self):
        # Local projection preserves branch identity through the figure-eight crossing.
        arc, self.segment = project_arc_length(
            self.points, self.lengths, self.data.qpos[:2], self.segment, window=4, closed=True
        )
        total = self.lengths[-1]
        delta = float((arc - self.arc + total / 2) % total - total / 2)
        self.arc = arc
        self.progress += delta
        fraction = (arc - self.lengths[self.segment]) / (
            self.lengths[self.segment + 1] - self.lengths[self.segment]
        )
        projection = self.points[self.segment] + fraction * (
            self.points[self.segment + 1] - self.points[self.segment]
        )
        return delta, float(np.linalg.norm(self.data.qpos[:2] - projection))

    def render(self):
        if self.render_mode == "human":
            import mujoco.viewer

            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.lookat[:] = [0, 0, 0]
                self.viewer.cam.distance = 2.8
                self.viewer.cam.elevation = -75
            if self.viewer.is_running():
                self.viewer.sync()
        return self.frame.copy()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.renderer.close()
