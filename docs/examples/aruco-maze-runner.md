# ArUco Maze Runner (PID)

This example drives the rover through a sequence of ArUco tags in order, visiting each one and stopping a set distance away, using a "stop-and-look-to-turn, then PID-drive-to-approach" strategy that's more careful than the single-tag pose-movement example about the camera's frame lag.

## Prerequisites

- A rover with the ESP32 camera reachable at `http://192.168.50.123:80/capture` (hardcoded as `img_addr`)
- The rover's UDP control endpoint reachable at `192.168.50.223` by default (or set `ROVER_IP`), plus a working `EncoderPoller` reply path on UDP port 9001 for wheel-encoder and lidar readings — don't run another script that binds that port at the same time
- `uv sync` already run from the repo root (installs `opencv-python`, `numpy`, and the workspace's `ArUco_detector` package)
- Two ArUco tags from `DICT_4X4_250`, printed at the default 10 cm marker size, with **IDs 0 and 1** — that's the `MARKER_ID_LIST` order this script drives
- A `logs/` directory writable from the repo (created automatically by `RunLogger`) since `LOG = True` writes a per-tick diagnostic CSV
- A display for `cv2.imshow`, since the script opens a window and quits on `Q` (or Ctrl-C)

## Run it

```bash
uv run rover_control/examples/aruco_maze_runner.py
```

No CLI arguments are read — the script's `if __name__ == "__main__":` block just constructs `ArucoRunner()` and calls `.update()`. Once running, the rover will turn to search for tag 0, center on it with a stop-look-turn cycle, drive to it with a PID controller, then repeat for tag 1, printing progress and writing a CSV log under `logs/` for each decision it makes.

## How it works

The class docstring explains the core problem this script solves: the ESP32 camera's images lag the rover's real heading by 1-2 frames, so if you decide where to turn while a frame from mid-turn is still in flight, you'll over-rotate. The fix is splitting each tag into a "stop and confirm" centering phase and a separate driving phase.

```python
class ArucoRunner(Rover):
    """
    Drive a list of ArUco tags in order using a PID, made robust to the camera's 1-2 frame lag.

    The ESP32 camera reports an image 1-2 frames behind the rover's real heading, so deciding
    while rotating over-rotates. We split each tag into two phases:
      1. CENTER with a stop-look-turn cycle - every reading is taken while stopped, after
         flushing the stale in-flight frames, so the lag can't trick us into over-rotating.
      2. DRIVE straight in with the PID. Because we only start once a confirmed (stopped)
         reading says the tag is centered, the heading lag during a straight approach is small;
         a guard stops and re-centers if the tag drifts off-center or drops out of view.
    """
```

A long list of class constants tunes exactly how cautious the centering process is — how close counts as "centered," how many frames must agree before a reading is trusted, and how far the rover sweeps when it can't see the tag at all. `MAX_VELOCITY` is also overridden lower than the base `Rover` class, because at full speed the chassis vibrates enough to lose tag lock.

```python
    MARKER_ID_LIST: ClassVar[list[int]] = [0, 1]  # the order of ArUco tag IDs we drive toward
    CENTER_TOLERANCE_DEG = (
        7.0  # "semi-centered" enough to drive; also keeps each turn above a tiny-burst size
    )
    DRIVE_RECENTER_DEG = (
        20.0  # only bail to re-center on big drift (tag nearing the frame edge), not small errors
    )
    SEARCH_STEP_DEG = 45.0  # how far we turn to sweep for a tag we can't see
    SETTLE_S = 0.25  # pause after a turn so the chassis stops moving before we look
    CONFIRM_FRAMES = 7  # fresh frames (captured after we stopped) sampled per look
    CONFIRM_MIN = 2  # tag must appear in at least this many fresh frames to trust the reading
    BEARING_AGREE_DEG = (
        5.0  # fresh frames must agree within this bearing spread, or we sample again
    )
    MAX_LOOK_TRIES = 5  # sampling rounds before we give up and treat the tag as unsteady
    LOG = True  # write a per-tick diagnostic CSV to logs/ (set False to disable)
    MAX_VELOCITY = 0.12  # m/s (overrides Rover.MAX_VELOCITY = 0.25)
```

Beyond the pose estimator and the two PID controllers (distance and heading, same idea as the other ArUco examples), the constructor also spins up an `EncoderPoller` — a background thread that keeps polling the rover for real wheel-encoder counts and lidar distances over UDP, used here purely for diagnostic logging, not for the actual control decisions.

```python
        # Diagnostic logging (one CSV row per tick/decision) + per-tick scratch from
        # compute_wheel_speeds
        self.logger = RunLogger("maze_pid", enabled=self.LOG)
        self._last_frame_stamp = 0.0
        self._dbg = {}

        # Background encoder + lidar poller - gives us actual wheel counts and proximity readings
        self.poller = EncoderPoller().start()
```

`turn_in_place` is the "look" phase's turning primitive: it spins the rover open-loop (timed, not measured) by a given number of degrees, based on how fast the wheels rotate the chassis at the motor's stall-floor speed.

```python
    def turn_in_place(self, direction, degrees, target_id=None):
        # Spin the rover on the spot by a number of degrees (open-loop, timed at the stall-floor
        # speed)
        # direction: +1 turns toward the tag's right (+X), -1 turns left
        omega = 2.0 * self.MIN_VELOCITY / self.wheel_separation
        spin_time = np.radians(degrees * self.TURN_SCALE) / omega
        spin = conversions.convert_linear_vel_to_angular_vel(
            self.MIN_VELOCITY, self.wheel_diameter / 2.0
        )

        # Turn right -> left wheel forward, right wheel backward (and the reverse for left)
        left, right = (spin, -spin) if direction > 0 else (-spin, spin)

        # Keep the spin command flowing at the firmware rate for the whole burst
        end_time = time.perf_counter() + spin_time
        while time.perf_counter() < end_time:
            self.write_real_velocities([left, right])
            self.sleep_to_command_rate()

        # Stop and settle so the next picture isn't motion-blurred
        self.stop()
        time.sleep(self.SETTLE_S)
```

`look()` is the "stop and stare" measurement at the heart of the lag-proofing: after stopping, it deliberately throws away any frame the camera captured before this exact moment (those are stale, from before/during the turn), then samples several *fresh* frames and only trusts the reading if enough of them saw the tag and their bearing estimates agree with each other.

```python
    def look(self, target_id):
        # Stop-and-stare measurement, hardened: trust a reading only when several fresh frames
        # AGREE.
        total_sampled = total_with_tag = 0

        for _ in range(self.MAX_LOOK_TRIES):
            since = time.perf_counter()  # ignore any frame the camera captured before right now
            bearings, distances = [], []
            for _ in range(self.CONFIRM_FRAMES):
                frame, since = self.estimator.next_frame(since)
                if frame is None:
                    break  # no fresh frame in time (brief camera hiccup) -> abandon this round
                total_sampled += 1
                _, poses = self.estimator.detect(frame)
                self._show(frame)
                if target_id in poses:
                    total_with_tag += 1
                    x, z = poses[target_id]["position"][0], poses[target_id]["position"][2]
                    bearings.append(
                        np.degrees(np.arctan2(x, z))
                    )  # bearing off straight-ahead (+ = right)
                    distances.append(z)  # Z = forward distance (meters)

            # Trust it only if enough fresh frames saw the tag AND they agree on the bearing
            if (
                len(bearings) >= self.CONFIRM_MIN
                and (max(bearings) - min(bearings)) <= self.BEARING_AGREE_DEG
            ):
                self._look_stats = {
                    "frames_sampled": total_sampled,
                    "frames_with_tag": total_with_tag,
                }
                return float(np.mean(bearings)), float(np.mean(distances))
            # Frames disagreed or too few sightings -> sample another fresh batch

        # Never got a steady, agreeing reading
        self._look_stats = {"frames_sampled": total_sampled, "frames_with_tag": total_with_tag}
        return None
```

`center()` uses `look()` in a loop: if the tag isn't steadily visible, sweep and look again; if it's visible but off to one side, turn exactly that amount and re-confirm; once a confirmed reading says the tag is within tolerance, the rover is considered centered and ready to drive.

```python
    def center(self, target_id):
        # Turn-look-turn until the tag is centered on a CONFIRMED reading.
        while True:
            reading = self.look(target_id)

            # Tag not steadily in view: sweep one step and look again
            if reading is None:
                self.turn_in_place(+1, self.SEARCH_STEP_DEG, target_id=target_id)
                continue

            bearing_deg, distance_m = reading

            # Off-center on a trustworthy reading: turn once by that bearing, then confirm again
            if abs(bearing_deg) > self.CENTER_TOLERANCE_DEG:
                self.turn_in_place(np.sign(bearing_deg), abs(bearing_deg), target_id=target_id)
                continue

            # Centered and confirmed - include lidar as an independent distance cross-check
            return
```

Once centered, `drive_to_tag()` takes over and behaves like the single-tag PID example — reading the freshest frame each tick, smoothing the tag's position, and computing wheel speeds — but it also watches for the tag drifting too far off-center or disappearing mid-approach, in which case it bails out and reports "recenter" instead of blindly continuing.

```python
    def drive_to_tag(self, tag_id, stop_distance_m=None):
        # Continuous PID approach toward an already-centered tag. Returns "arrived" once we're
        # within the stop tolerance, or "recenter" if the tag drifts off-center or drops out of
        # view.
        if stop_distance_m is not None:
            self.stop_tolerance_m = stop_distance_m

        # Start the controllers clean so the first dt isn't a stale stop-and-stare gap
        self.distance_pid.reset()
        self.heading_pid.reset()
        self._last_pid_time = time.perf_counter()
        _last_known_distance = None  # used to detect "tag went below FOV when very close"

        while True:
            frame, stamp = self.estimator.camera.latest()
            annotated, poses = self.estimator.detect(frame)
            self._show(annotated if annotated is not None else frame)

            seen = tag_id in poses
```

`update()` is the top-level orchestration: for each tag in `MARKER_ID_LIST`, keep alternating between centering and driving until `drive_to_tag` actually reports "arrived" (rather than bailing to recenter), then move to the next tag. The `finally` block guarantees the rover stops and every background resource (camera, encoder poller, logger, window) shuts down cleanly even on Ctrl-C.

```python
    def update(self):
        try:
            # Visit each tag in order
            for target_id in self.MARKER_ID_LIST:
                print(f"Hunting tag {target_id}...")

                # Keep (re)centering and driving until we actually arrive
                while True:
                    self.center(target_id)  # stop-stare until confirmed centered
                    if self.drive_to_tag(target_id) == "arrived":
                        break  # otherwise we drifted/lost it -> re-center
                print(f"Reached tag {target_id}!")

            print("Maze complete - every tag in the list visited!")
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            # Make sure the rover stops, the camera + log close, and the window closes
            self.stop()
            self.estimator.close()
            self.poller.stop()
            self.logger.close()
            cv2.destroyAllWindows()
```

## See also

- [rover_control/](../../rover_control/) — the Rover base class and other examples
- [ArUco_detector/](../../ArUco_detector/) — the marker detection module this example depends on
- [Back to README](../../README.md)
