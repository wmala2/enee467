# ArUco Maze Runner (Trapezoid)

This example drives the same two-tag maze as the PID maze runner, but instead of continuously re-measuring the tag while driving, it measures once while stopped, plans a whole speed curve (ramp up, cruise, ramp down) in advance, and drives that plan open-loop — trading mid-course correction for immunity to motion-blur tag loss at higher speed.

## Prerequisites

- A rover with the ESP32 camera reachable at `http://192.168.50.123:80/capture` (hardcoded as `img_addr`)
- The rover's UDP control endpoint reachable at `192.168.50.223` by default (or set `ROVER_IP`), plus a working `EncoderPoller` reply path on UDP port 9001 for wheel-encoder and lidar readings — don't run another script that binds that port at the same time
- `uv sync` already run from the repo root (installs `opencv-python`, `numpy`, and the workspace's `ArUco_detector` package)
- Two ArUco tags from `DICT_4X4_250`, printed at the default 10 cm marker size, with **IDs 0 and 1** — that's the `MARKER_ID_LIST` order this script drives
- A `logs/` directory writable from the repo (created automatically by `RunLogger`) since `LOG = True` writes a per-decision diagnostic CSV
- A flat, obstacle-free path between tags, since the drive itself is open-loop and cannot react to the camera mid-move
- A display for `cv2.imshow`, since the script opens a window and quits on `Q` (or Ctrl-C)

## Run it

```bash
uv run rover_control/examples/aruco_maze_runner_trapezoid.py
```

No CLI arguments are parsed — the `if __name__ == "__main__":` block just constructs `ArucoTrapezoidRunner()` and calls `.update()`. Once running, the rover searches for tag 0, centers on it with a stop-look-turn cycle, then drives a pre-planned ramp-up/cruise/ramp-down speed curve straight at it without looking, verifies it actually arrived, and repeats for tag 1.

## How it works

The docstring lays out the design trade-off directly: this script uses the same stop-and-stare centering idea as the PID maze runner, but replaces the "watch the camera the whole way in" approach with "measure once, then commit to a planned move."

```python
class ArucoTrapezoidRunner(Rover):
    """
    Same maze goal as aruco_maze_runner.py, but a different control idea:
    instead of a PID reacting to every (laggy) camera frame, we STOP, LOOK, then MOVE.

    The ESP32 camera reports an image 1-2 frames behind the rover's real heading, so any
    decision made while turning is based on a stale view and over-rotates. To beat that, every
    measurement is taken while stopped: we discard the stale in-flight frames, average a few
    fresh ones, and only then decide to turn again or drive. For each tag we (1) turn-look-turn
    until it is centered on a CONFIRMED reading, then (2) drive a trapezoidal speed curve
    (ramp up, cruise, ramp down) straight to it - taking no pictures while moving.
    """
```

The class constants tune the trapezoid shape (top speed and acceleration) as well as how strict the centering and verification steps are. Note there's no PID here at all — this script is built around timed motion, not continuous feedback.

```python
    MARKER_ID_LIST: ClassVar[list[int]] = [0, 1]  # the order of ArUco tag IDs we drive toward
    CRUISE_VELOCITY = 0.20  # top speed of the trapezoid (m/s), under the rover's MAX_VELOCITY
    ACCEL = 0.20  # how fast the speed ramps up and down (m/s^2)
    CENTER_TOLERANCE_DEG = (
        7.0  # "semi-centered" enough to drive; also keeps each turn above a tiny-burst size
    )
    SEARCH_STEP_DEG = 75.0  # how far we turn to sweep for a tag we can't see
    SETTLE_S = 0.25  # pause after a turn so the chassis stops moving before we look
    CONFIRM_FRAMES = 5  # fresh frames (captured after we stopped) sampled per look
    CONFIRM_MIN = 2  # tag must appear in at least this many fresh frames to trust the reading
    BEARING_AGREE_DEG = (
        5.0  # fresh frames must agree within this bearing spread, or we sample again
    )
    MAX_LOOK_TRIES = 4  # sampling rounds before we give up and treat the tag as unsteady
    GOAL_TOLERANCE_M = 0.05  # how close beyond the stop distance still counts as "reached"
    MAX_APPROACH_TRIES = 5  # drive + re-verify attempts per tag (we drive open-loop, so we check)
    LOG = True  # write a per-decision diagnostic CSV to logs/ (set False to disable)
```

`turn_in_place` and `look` are essentially identical in spirit to the PID maze runner's versions: an open-loop timed spin, and a "stop, discard stale frames, sample fresh ones, only trust readings multiple frames agree on" measurement.

```python
    def turn_in_place(self, direction, degrees, target_id=None):
        # Spin the rover on the spot by a number of degrees (open-loop, timed at the stall-floor
        # speed)
        # direction: +1 turns toward the tag's right (+X), -1 turns left
        omega = 2.0 * self.MIN_VELOCITY / self.wheel_separation
        spin_time = np.radians(degrees) / omega
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

`approach_tag()` is the turn-look-turn centering loop (same structure as the PID runner's `center()`), but here it also returns the confirmed distance to the tag, since that distance is what the trapezoid planner needs next.

```python
    def approach_tag(self, target_id):
        # Turn-look-turn until the tag is centered on a CONFIRMED reading, then return its distance.
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
            return distance_m
```

`drive_straight()` is where the "trapezoid" name comes from: a speed profile that ramps up from the motor's stall floor to cruise speed, holds cruise, then ramps back down, so the rover accelerates and decelerates smoothly instead of slamming to a stop. It first works out how much distance one ramp covers, and falls back to a lower-peaked triangle if the total trip is too short to ever reach cruise speed.

```python
    def drive_straight(self, distance_m, target_id=None):
        # Play back a trapezoidal speed curve over time to cover distance_m, both wheels equal
        # (straight)
        v_min = self.MIN_VELOCITY  # motors stall below this, so the ramps start/end here, not at 0
        v_cruise = self.CRUISE_VELOCITY
        a = self.ACCEL

        # One ramp goes from v_min up to v_cruise; work out its time and the distance it covers
        t_ramp = (v_cruise - v_min) / a
        d_ramp = (v_min + v_cruise) / 2.0 * t_ramp

        # If the trip is too short to ever reach cruise speed, use a triangle (lower peak speed)
        if 2.0 * d_ramp >= distance_m:
            v_top = min(v_cruise, np.sqrt(v_min**2 + a * distance_m))
            t_ramp = (v_top - v_min) / a
            t_cruise = 0.0
        else:
            v_top = v_cruise
            t_cruise = (distance_m - 2.0 * d_ramp) / v_cruise

        # Total time the curve takes: ramp up + cruise + ramp down
        t_total = 2.0 * t_ramp + t_cruise
```

With the plan computed, the actual driving is just a loop that figures out which part of the trapezoid the current instant falls into (ramping up, cruising, or ramping down) and sends the matching speed to both wheels every tick, since driving straight means both wheels move at the same rate.

```python
        # Send the matching speed every command tick until the curve finishes
        start = time.perf_counter()
        while True:
            t = time.perf_counter() - start
            if t >= t_total:
                break

            # Pick this instant's speed from whichever part of the trapezoid we're in
            if t < t_ramp:
                v = v_min + a * t  # ramping up
            elif t < t_ramp + t_cruise:
                v = v_top  # cruising
            else:
                v = v_top - a * (t - t_ramp - t_cruise)  # ramping down

            # Never command below the stall floor, then send both wheels the same (straight) speed
            v = max(v, v_min)
            wheel = conversions.convert_linear_vel_to_angular_vel(v, self.wheel_diameter / 2.0)
            self.write_real_velocities([wheel, wheel])
            self.sleep_to_command_rate()

        # Cut to zero at the end of the planned move
        self.stop()
```

Because this whole drive is open-loop (no camera feedback while moving, no encoder-based distance tracking), it can under- or overshoot. `reach_tag()` compensates by looking again after each planned drive and, if it's not yet within tolerance, computing the leftover distance and driving that gap too — repeating up to `MAX_APPROACH_TRIES` times.

```python
    def reach_tag(self, target_id):
        # Center, drive, then VERIFY we arrived. We drive open-loop (no encoders), so one trapezoid
        # can under/overshoot - take fresh captures afterward and re-drive the re-measured gap if we
        # fell short.
        distance = self.approach_tag(target_id)  # initial search + center + confirmed distance
        for attempt in range(self.MAX_APPROACH_TRIES):
            remaining = distance - self.stop_tolerance_m
            if remaining <= self.GOAL_TOLERANCE_M:
                return True  # confirmed within reach

            # Drive the measured gap, then look again to check we actually got there
            self.drive_straight(remaining, target_id)
            reading = self.look(target_id)

            # Tag no longer in view after driving its distance -> we're at (or past) it
            if reading is None:
                return True

            # Still see it: update the distance, and re-aim if the blind drive left us off-center
            bearing_deg, distance = reading
            if abs(bearing_deg) > self.CENTER_TOLERANCE_DEG:
                self.turn_in_place(np.sign(bearing_deg), abs(bearing_deg), target_id=target_id)

        return False  # ran out of tries without confirming
```

`update()` ties it together, visiting each tag with `reach_tag()` and reporting whether it was confirmed or given up on, then cleaning up the rover, camera, logger, and window in the `finally` block even if the run is interrupted.

```python
    def update(self):
        try:
            # Visit each tag in order: center, drive the planned curve, then verify we arrived
            for target_id in self.MARKER_ID_LIST:
                print(f"Looking for tag {target_id}...")
                if self.reach_tag(target_id):
                    print(f"Reached tag {target_id} (verified)!")
                else:
                    print(
                        f"Could not confirm tag {target_id} after {self.MAX_APPROACH_TRIES} tries "
                        f"- moving on."
                    )

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
