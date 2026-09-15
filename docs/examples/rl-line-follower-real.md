# RL Line Follower (Real)

This example runs a PPO policy — trained in `rover_mujoco` and evaluated against the deployment contract — on the physical rover, closing the sim-to-real loop for line following: the same eleven-input observation, two-action policy that drives the simulated rover now reads the real onboard camera and wheel encoders.

## Prerequisites

- A qualified PPO checkpoint (`model.zip`) and a matching deployment **manifest** JSON produced by `rover_mujoco/scripts/publish_policy.py` — `RLLineFollowerRover.__init__` calls `check_manifest`, which refuses to start if the manifest's recorded SHA-256 or API version doesn't match the model file.
- The physical rover reachable over UDP (`--rover-addr`, defaults to `$ROVER_IP` or the built-in default address) and its ESP32 camera reachable over HTTP (`--camera-addr`).
- `stable-baselines3` (for `PPO.load`), `opencv-python` (`cv2`, used to resize camera frames), and the `ArUco_detector` package's `CameraStream` — all installed by the same root-level `uv sync` used for the simulator.
- Known encoder calibration for your physical rover — `--counts-per-revolution` and `--encoder-signs` default to the firmware's own constants (`Rover.ENCODER_COUNTS_PER_REVOLUTION = 680.0`, `Rover.ENCODER_SIGNS = (1, 1)`), so only override them if your rover was built differently.

## Run it

```bash
uv run rover_control/examples/rl_line_follower.py <path/to/model.zip> <path/to/manifest.json>
```

The rover starts driving using the policy's predicted wheel commands, printing "Running the qualified PPO line follower; Ctrl-C stops the rover." — Ctrl-C (or any sensor/inference error) stops the motors and cleanly shuts down the camera and encoder connections.

## How it works

`main` parses the two required paths (model, manifest) plus optional hardware-calibration overrides, then constructs an `RLLineFollowerRover` — construction itself performs the safety check (manifest match, and a dry-run of the encoder math) *before* opening any network connection to the actual hardware.

```python
    # Construction verifies simulation evidence before any hardware connection starts.
    rover = RLLineFollowerRover(
        args.model,
        args.manifest,
        counts_per_revolution=args.counts_per_revolution,
        encoder_signs=args.encoder_signs,
        wheel_diameter_m=2 * args.wheel_radius_m,
        camera_addr=args.camera_addr,
        rover_addr=args.rover_addr,
        record_dir=args.record,
        show_camera=True,
    )
```

`RLLineFollowerRover.__init__` (in `rover_control/rl_rover.py`) validates the manifest and encoder math, loads the PPO model onto the CPU, and rejects any checkpoint that doesn't have the exact observation/action shape the deployment interface expects — a policy trained for a different task simply cannot be loaded here.

```python
        check_manifest(model_path, manifest_path)
        encoder_speeds([0, 0], 0.1, counts_per_revolution, encoder_signs)
        self.model = PPO.load(model_path, device="cpu")
        if self.model.observation_space.shape != (11,) or self.model.action_space.shape != (2,):
            raise ValueError("This runner requires the eleven-input, two-action PPO policy")
```

The real world doesn't hand you clean, synchronous sensor data the way simulation does — a background thread polls the wheel encoders over UDP at the same 10 Hz the policy was trained at, tracking how much each wheel has turned since the last poll (and how much real time that took, since the network doesn't guarantee exact timing).

```python
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
```

`_get_obs` is the hardware equivalent of the simulator's `_get_observation`: it grabs the latest camera frame and encoder delta, and — critically — refuses to hand the policy stale data. If either sensor hasn't updated recently enough, it raises instead of letting the rover drive blind on old information.

```python
        frame, stamp = self._camera.latest()
        counts, elapsed, encoder_stamp = self._encoders.latest()
        now = time.perf_counter()
        # Hold zero during sensor startup and stop on stale data once the startup period ends.
        if frame is None or elapsed <= 0 or now - min(stamp, encoder_stamp) > self.STALE_LIMIT_S:
            if now - self._started < 3:
                return None
```

Once it has a fresh frame and encoder reading, it converts them into exactly the same 11-value observation vector the simulated policy was trained on (`observation_from_sensors`, shared code with the sim environment) — this shared function is what makes "trained in sim, run on hardware" actually work.

```python
        image = cv2.resize(frame, (CAM_RES, CAM_RES))[:, :, ::-1]
        speeds = encoder_speeds(counts, elapsed, self._counts_per_revolution, self._encoder_signs)
        observation = observation_from_sensors(image, speeds)
```

`compute_wheel_speeds` is called every control tick: it builds the observation, checks whether the line has been lost for too long (stopping the rover rather than driving off blind), asks the policy for an action, and converts that normalized action into wheel speeds clipped to what the real motors can safely do.

```python
    def compute_wheel_speeds(self):
        observation = self._get_obs()
        if observation is None:
            return np.zeros(2)
        self._lost_steps = 0 if observation[2] else self._lost_steps + 1
        if self._lost_steps >= self.LINE_LOST_STOP_STEPS:
            raise RuntimeError(f"Line lost for {self.LINE_LOST_STOP_STEPS / CONTROL_HZ:.1f} s")
        action, _ = self.model.predict(observation, deterministic=True)
        targets = wheel_targets(action)
        radius = self.wheel_diameter / 2
        targets = np.clip(targets, -self.MAX_VELOCITY / radius, self.MAX_VELOCITY / radius)
        return np.where(np.abs(targets) < self.MIN_VELOCITY / radius, 0.0, targets)
```

## See also

- [rover_mujoco/](../../rover_mujoco/) — the simulation environment this example depends on
- [rover_control/](../../rover_control/) — the physical-rover driver this example runs on
- [Back to README](../../README.md)
