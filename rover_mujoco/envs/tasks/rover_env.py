import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class RoverEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        model_path = os.path.join(
            os.path.dirname(__file__), "../../assets/robots/rover/rover_scene.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # 2D action space: [left_wheel_ctrl, right_wheel_ctrl] target angular velocities (rad/s)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

        # Observation space: base position (3) + orientation quat (4) + linear vel (3) + angular vel (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # Drive the left/right wheel velocity actuators directly
        self.data.ctrl[:] = action
        # Advance physics by one timestep using the actuator forces above
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # base_link's free joint occupies qpos[0:7] (pos + quat) and qvel[0:6] (linear + angular vel)
        pos_quat = self.data.qpos[:7]
        vel = self.data.qvel[:6]
        return np.concatenate([pos_quat, vel]).astype(np.float32)
