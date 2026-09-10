"""Run the current eleven-input PPO policy after offline simulation qualification."""

import json
from pathlib import Path
import socket
import threading
import time

from envs.contract import observation_from_sensors
from envs.contract import wheel_targets
import numpy as np
from stable_baselines3 import PPO

from ArUco_detector.camera_stream import CameraStream
from rover_control import network_interface
from rover_control.ppo_deployment import check_manifest
from rover_control.ppo_deployment import encoder_speeds
from rover_control.rover import Rover

CAM_RES = 64  # must match the trained policy's observation size
CONTROL_HZ = 10.0  # matches Rover.COMMAND_RATE_HZ and the policy's training rate


class _EncoderOnlyPoller:
    """Poll encoder counts at 10 Hz and retain their actual measurement interval."""

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
        self._stamp = 0.0
        self._elapsed = 0.0
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
                left, right = float(parsed["left_encoder"]), float(parsed["right_encoder"])
                if not np.isfinite([left, right]).all():
                    continue
                stamp = time.perf_counter()
                with self._lock:
                    if self._prev_left is not None and self._prev_right is not None:
                        self._delta = np.array(
                            [left - self._prev_left, right - self._prev_right], dtype=np.float32
                        )
                        self._elapsed = stamp - self._stamp
                    self._prev_left, self._prev_right = left, right
                    self._stamp = stamp
            except (TimeoutError, ValueError, KeyError, TypeError):
                pass  # The unchanged timestamp lets the control loop detect stale readings.

            elapsed = time.perf_counter() - t0
            remaining = self._interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def latest(self):
        with self._lock:
            return self._delta.copy(), self._elapsed, self._stamp


class RLLineFollowerRover(Rover):
    """Use calibrated encoder units and the exact normalized action mapping seen in training."""

    # Control steps of continuous line loss before the rover stops. Simulation terminates at
    # half that; see compute_wheel_speeds for why hardware is given longer.
    LINE_LOST_STOP_STEPS = 15

    # How old either sensor reading may be before the rover stops.
    #
    # 0.25 s sat inside the camera's own latency tail rather than outside it. Measured over
    # 356 steps of runs that completed: median 28 ms, p99 167, max 232, with 0.28% of frames
    # past 230 ms. At 10 Hz that is a near-miss roughly every 36 seconds, so runs did not fail
    # because something was wrong, they failed because they lasted long enough. Three stalls
    # all reported 0.26-0.27 s with the encoder healthy at 0.07-0.10.
    #
    # One second is well clear of that tail while still stopping promptly on a camera that has
    # actually died. The cost is bounded: at MAX_VELOCITY the rover covers 25 cm on a stale
    # image, the same exposure already accepted for line loss.
    STALE_LIMIT_S = 1.0

    # Camera polling rate. Kept above CONTROL_HZ so a fresh frame is normally waiting rather
    # than being fetched on demand; the stream's own comment notes load and heat as the reason
    # not to raise it without cause.
    CAMERA_POLL_HZ = 12.5

    # Class-level default so the attribute exists even when a caller builds the object without
    # running __init__, as the deployment tests do to exercise the conversion path offline.
    _recorder = None

    def __init__(
        self,
        model_path,
        manifest_path,
        *,
        counts_per_revolution,
        encoder_signs,
        camera_addr=Rover.DEFAULT_CAMERA_ADDR,
        record_dir=None,
        **kwargs,
    ):
        # Validate local evidence and calibration before opening camera or encoder connections.
        check_manifest(model_path, manifest_path)
        encoder_speeds([0, 0], 0.1, counts_per_revolution, encoder_signs)
        self.model = PPO.load(model_path, device="cpu")
        if self.model.observation_space.shape != (11,) or self.model.action_space.shape != (2,):
            raise ValueError("This runner requires the eleven-input, two-action PPO policy")
        kwargs.setdefault("wheel_diameter_m", 2 * 0.03435)
        super().__init__(camera_addr=camera_addr, **kwargs)
        self._counts_per_revolution = counts_per_revolution
        self._encoder_signs = encoder_signs
        # Poll faster than the control loop consumes. Fetching at exactly CONTROL_HZ left no
        # headroom: one slow HTTP GET makes the newest frame older than a control step, and a
        # second in a row exceeded STALE_LIMIT_S and stopped the rover mid-drive.
        self._camera = CameraStream(camera_addr, target_hz=self.CAMERA_POLL_HZ)
        self._encoders = _EncoderOnlyPoller(poll_hz=CONTROL_HZ)
        self._started = time.perf_counter()
        self._lost_steps = 0
        self._recorder = _RunRecorder(record_dir) if record_dir else None
        try:
            self._camera.start()
            self._encoders.start()
        except Exception:
            self.close()
            raise

    def _get_obs(self):
        import cv2

        frame, stamp = self._camera.latest()
        counts, elapsed, encoder_stamp = self._encoders.latest()
        now = time.perf_counter()
        # Hold zero during sensor startup and stop on stale data once the startup period ends.
        if frame is None or elapsed <= 0 or now - min(stamp, encoder_stamp) > self.STALE_LIMIT_S:
            if now - self._started < 3:
                return None
            # Name the sensor and the age. One message for three unrelated causes made a
            # hardware stall undiagnosable, and these fail for entirely different reasons:
            # no frame at all, an encoder reply that never came, or a link gone quiet.
            faults = []
            if frame is None:
                faults.append("no camera frame received")
            else:
                faults.append(f"camera frame {now - stamp:.2f}s old")
            if elapsed <= 0:
                faults.append("no encoder reply since the last poll")
            else:
                faults.append(f"encoder reading {now - encoder_stamp:.2f}s old")
            raise RuntimeError(
                f"Sensors stale after {now - self._started:.1f}s "
                f"(limit {self.STALE_LIMIT_S:.2f}s): " + "; ".join(faults)
            )
        self._maybe_show(frame)
        image = cv2.resize(frame, (CAM_RES, CAM_RES))[:, :, ::-1]
        speeds = encoder_speeds(counts, elapsed, self._counts_per_revolution, self._encoder_signs)
        observation = observation_from_sensors(image, speeds)
        if self._recorder is not None:
            self._recorder.write(image, observation, now - stamp, now - encoder_stamp)
        return observation

    def compute_wheel_speeds(self):
        observation = self._get_obs()
        if observation is None:
            return np.zeros(2)
        # Deliberately looser than simulation's half second, which the policy was trained
        # against: a real camera drops frames in ways the renderer never did, and stopping on
        # a transient is its own failure. The cost is blind distance, MAX_VELOCITY * this, so
        # about 25 cm. Tighten it back toward simulation once the logs show how often it fires.
        self._lost_steps = 0 if observation[2] else self._lost_steps + 1
        if self._lost_steps >= self.LINE_LOST_STOP_STEPS:
            raise RuntimeError(f"Line lost for {self.LINE_LOST_STOP_STEPS / CONTROL_HZ:.1f} s")
        action, _ = self.model.predict(observation, deterministic=True)
        targets = wheel_targets(action)
        radius = self.wheel_diameter / 2
        targets = np.clip(targets, -self.MAX_VELOCITY / radius, self.MAX_VELOCITY / radius)
        return np.where(np.abs(targets) < self.MIN_VELOCITY / radius, 0.0, targets)

    def update(self):
        # Stop on sensor failure, inference error, or keyboard interruption.
        try:
            super().update()
        finally:
            self.stop()

    def close(self):
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        self._encoders.stop()
        self._camera.stop()
        super().close()


class _RunRecorder:
    """Save what the detector sees, not what the rover looks like.

    A video of the robot cannot show why a frame failed detection. This writes the same 64x64
    image the policy is given, upscaled, with the detection mask painted over it and the
    eleven observation values beside it, plus a CSV of those values. Recording never
    interrupts driving: any failure disables the recorder and the run continues.
    """

    SCALE = 8

    def __init__(self, directory):
        import csv

        self._dir = Path(directory)
        # Never overwrite: a failed run's log is the evidence for why it failed, and reusing
        # the directory once destroyed exactly that.
        if self._dir.exists() and any(self._dir.iterdir()):
            stamp = time.strftime("%H%M%S")
            self._dir = self._dir.with_name(f"{self._dir.name}-{stamp}")
        self._dir.mkdir(parents=True, exist_ok=True)
        print(f"recording to {self._dir}")
        self._writer = None
        self._video = None
        # Held open for the run's lifetime and closed in close(); a context manager here
        # would shut the file before the first frame is written.
        self._rows = open(  # noqa: SIM115
            self._dir / "observations.csv", "w", newline="", encoding="utf-8"
        )
        self._csv = csv.writer(self._rows)
        self._csv.writerow(
            ["t"]
            + [f"{band}_{field}" for band in ("near", "mid", "far") for field in ("x", "y", "seen")]
            + ["left_rad_s", "right_rad_s", "min_rgb", "camera_age_s", "encoder_age_s"]
        )
        self._start = time.perf_counter()

    def write(self, image, observation, camera_age=float("nan"), encoder_age=float("nan")):
        try:
            import cv2
            from envs.pid import centroid_error

            _, mask = centroid_error(image, band=1.0)
            view = cv2.resize(
                image[:, :, ::-1],
                (CAM_RES * self.SCALE, CAM_RES * self.SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            painted = cv2.resize(
                mask.astype(np.uint8) * 255,
                (CAM_RES * self.SCALE, CAM_RES * self.SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            view[painted > 0] = (0, 0, 255)
            elapsed = time.perf_counter() - self._start
            # The darkest pixel present is the number that decides whether the tape is found
            # at all, since detection needs every channel under its threshold.
            darkest = int(image.reshape(-1, 3).min(axis=1).min())
            for index, text in enumerate((
                f"t={elapsed:6.2f}s  darkest_rgb={darkest:3d}",
                f"near x={observation[0]:+.2f} seen={observation[2]:.0f}",
                f"mid  x={observation[3]:+.2f} seen={observation[5]:.0f}",
                f"far  x={observation[6]:+.2f} seen={observation[8]:.0f}",
            )):
                cv2.putText(
                    view,
                    text,
                    (6, 16 + 16 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            if self._video is None:
                self._video = cv2.VideoWriter(
                    str(self._dir / "camera.mp4"),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    CONTROL_HZ,
                    (view.shape[1], view.shape[0]),
                )
            self._video.write(view)
            self._csv.writerow(
                [f"{elapsed:.3f}"]
                + [f"{v:.4f}" for v in observation]
                + [darkest, f"{camera_age:.3f}", f"{encoder_age:.3f}"]
            )
        except Exception as error:  # noqa: BLE001  driving must not stop for a recorder
            print(f"recording disabled: {error}")
            self.close()

    def close(self):
        if self._video is not None:
            self._video.release()
            self._video = None
        if self._rows is not None:
            self._rows.close()
            self._rows = None
