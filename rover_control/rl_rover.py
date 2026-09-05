"""RL-based line follower: drives the rover with a PPO policy trained in MuJoCo
(rover_mujoco, https://huggingface.co/CursedRock17/rover-line-follower-ppo) instead of the
classical ArUco/YOLO navigation the rest of this package provides.

Deployed as-is (no retraining around real motor limits) with clamping applied here instead
-- watch for jerky behavior right at the clamp boundaries; that's the known, accepted
trade-off of this choice. Sim<->real translation points this module handles, all found by
reconciling rover_mujoco's env against this package's already-working hardware interface:

- **Sign convention**: the policy was trained where equal-sign wheel commands spin the rover
  in place; Rover.write_real_velocities() sends left/right straight through, i.e. equal sign
  really is forward on this real interface. NEGATE_RIGHT undoes the sim's convention.
- **Units**: the policy outputs rad/s; write_real_velocities() converts to the firmware's
  m/s itself, so clamping happens in rad/s before calling it, not after.
- **Real velocity floor/ceiling**: the policy never saw Rover.MIN_VELOCITY/MAX_VELOCITY
  (the real motors' dead zone and top speed) during training -- clamped here at inference
  time instead.
- **Encoder ticks**: EncoderPoller alternates encoder/lidar queries, halving its effective
  encoder rate to 5Hz. The policy expects a fresh tick-delta every 10Hz control step
  (matching training), so this module polls encoders only, at the full rate.
"""
import json
import socket
import threading
import time

import numpy as np
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO

from ArUco_detector.camera_stream import CameraStream
from rover_control import network_interface
from rover_control.rover import Rover

HF_REPO_ID = "CursedRock17/rover-line-follower-ppo"
CAM_RES = 64                # must match the trained policy's observation size
CONTROL_HZ = 10.0           # matches Rover.COMMAND_RATE_HZ and the policy's training rate


class _EncoderOnlyPoller:
    """Polls {"command": "e"} every tick at the full control rate and returns ticks *since
    the last poll* -- exactly the observation the policy was trained on. Deliberately
    separate from encoder_poller.EncoderPoller (which alternates with a lidar query, halving
    the effective encoder rate) rather than modifying that shared, already-working class."""

    def __init__(self, poll_hz=CONTROL_HZ):
        self._interval = 1.0 / poll_hz
        self._query = json.dumps({"command": "e"}).encode("utf-8")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", network_interface.UDP_REPLY_PORT))
        self._sock.settimeout(self._interval * 0.8)
        self._lock = threading.Lock()
        self._prev_left = self._prev_right = None
        self._delta = np.zeros(2, dtype=np.float32)
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="rl_encoder_poller")
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _run(self):
        while self._running:
            t0 = time.perf_counter()
            network_interface.send_message(self._query)
            try:
                data, _ = self._sock.recvfrom(512)
                parsed = json.loads(data.decode("utf-8"))
                left, right = parsed["left_encoder"], parsed["right_encoder"]
                with self._lock:
                    if self._prev_left is not None:
                        self._delta = np.array(
                            [left - self._prev_left, right - self._prev_right], dtype=np.float32
                        )
                    self._prev_left, self._prev_right = left, right
            except (socket.timeout, ValueError, KeyError):
                pass  # missed packet: keep the last delta rather than stall the control loop

            elapsed = time.perf_counter() - t0
            remaining = self._interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def latest_delta(self):
        with self._lock:
            return self._delta.copy()


class RLLineFollowerRover(Rover):
    """Drives using the PPO line-following policy in place of a hand-written
    compute_wheel_speeds(). See this module's docstring for the sim<->real translation."""

    NEGATE_RIGHT = True  # undoes the sim's opposite-sign-for-forward convention

    def __init__(self, camera_addr=Rover.DEFAULT_CAMERA_ADDR, **kwargs):
        super().__init__(camera_addr=camera_addr, **kwargs)

        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.zip")
        # CPU, not "auto": one observation at 10Hz never needs a GPU, and "auto" has already
        # crashed here once on a GPU/CUDA-build mismatch (compute capability newer than this
        # machine's PyTorch build supports) -- sidestep that class of failure entirely.
        self.model = PPO.load(model_path, device="cpu")

        self._camera = CameraStream(camera_addr, target_hz=CONTROL_HZ).start()
        self._encoders = _EncoderOnlyPoller(poll_hz=CONTROL_HZ).start()

        # rad/s equivalents of the real motors' dead zone/top speed (MIN/MAX_VELOCITY are m/s)
        wheel_radius = self.wheel_diameter / 2.0
        self._min_omega = self.MIN_VELOCITY / wheel_radius
        self._max_omega = self.MAX_VELOCITY / wheel_radius

    def _get_obs(self):
        frame, _ = self._camera.latest()
        self._maybe_show(frame)
        if frame is None:
            image = np.zeros((CAM_RES, CAM_RES, 1), dtype=np.uint8)
        else:
            import cv2
            resized = cv2.resize(frame, (CAM_RES, CAM_RES))
            image = resized.mean(axis=-1, keepdims=True).astype(np.uint8)
        return {"image": image, "encoders": self._encoders.latest_delta()}

    def compute_wheel_speeds(self):
        action, _ = self.model.predict(self._get_obs(), deterministic=True)
        left, right = float(action[0]), float(action[1])
        if self.NEGATE_RIGHT:
            right = -right
        return self.clamp([left, right], self._min_omega, self._max_omega)

    def close(self):
        self._encoders.stop()
        self._camera.stop()
        super().close()
