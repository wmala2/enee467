# External Libraries
import time

import cv2
import numpy as np

# Local Libraries to Import
from ArUco_detector.aruco_pose_estimator import ArucoPoseEstimator

# Local Files to Import
from rover_control import conversions
from rover_control.encoder_poller import EncoderPoller
from rover_control.rover import Rover
from rover_control.run_logger import RunLogger


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

    MARKER_ID_LIST = [0, 1]  # the order of ArUco tag IDs we drive toward
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

    def __init__(self, stop_tolerance_m=0.25):
        super().__init__()

        # How far away from the tag (in meters) the rover should stop
        self.stop_tolerance_m = stop_tolerance_m

        # Camera Stream Address
        img_addr = "http://192.168.50.123:80/capture"

        # Create our ArUco Marker Pose Estimator
        self.estimator = ArucoPoseEstimator(
            http_addr=img_addr,
            verbose=True,  # set False if you only want data
        )

        # Diagnostic logging (one CSV row per decision)
        self.logger = RunLogger("maze_trap", enabled=self.LOG)

        # Background encoder + lidar poller - gives us actual wheel counts and proximity readings
        self.poller = EncoderPoller().start()

        # Populated by look() so approach_tag() can log detection reliability alongside its decisions
        self._look_stats = {}

    def _enc(self):
        # Return encoder columns as a dict for log() calls; empty dict if no reply yet
        data = self.poller.latest()
        if data is None:
            return {}
        l, r, dl, dr = data
        return {"enc_left": l, "enc_right": r, "enc_left_delta": dl, "enc_right_delta": dr}

    def _lidar(self):
        # Return lidar distance columns for log() calls; empty dict if no reply yet
        data = self.poller.lidar_latest()
        if data is None:
            return {}
        l, c, r, _ = data
        return {"lidar_left_mm": l, "lidar_center_mm": c, "lidar_right_mm": r}

    def _show(self, frame):
        # Draw the camera frame and quit the whole run if the user presses Q
        if frame is not None:
            cv2.imshow("Aruco Trapezoid", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def turn_in_place(self, direction, degrees, target_id=None):
        # Spin the rover on the spot by a number of degrees (open-loop, timed at the stall-floor speed)
        # direction: +1 turns toward the tag's right (+X), -1 turns left
        omega = 2.0 * self.MIN_VELOCITY / self.wheel_separation
        spin_time = np.radians(degrees) / omega
        spin = conversions.convert_linear_vel_to_angular_vel(
            self.MIN_VELOCITY, self.wheel_diameter / 2.0
        )

        # Snapshot encoder counts before the turn so we can measure actual rotation afterward
        enc_before = self.poller.latest()

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

        # Log the turn outcome: commanded angle vs total encoder delta (actual rotation proxy)
        enc_after = self.poller.latest()
        if enc_before is not None and enc_after is not None:
            self.logger.log(
                phase="turn",
                event="completed",
                target_id=target_id,
                commanded_deg=direction * degrees,
                enc_left_delta=enc_after[0] - enc_before[0],
                enc_right_delta=enc_after[1] - enc_before[1],
            )

    def look(self, target_id):
        # Stop-and-stare measurement, hardened: trust a reading only when several fresh frames AGREE.
        # next_frame() only returns frames captured AFTER `since`, so the stale, laggy frames from
        # while we were moving are skipped deterministically - no frame-count guessing.
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

    def approach_tag(self, target_id):
        # Turn-look-turn until the tag is centered on a CONFIRMED reading, then return its distance.
        # Every decision is made while stopped, so the camera lag can't trick us into over-rotating.
        while True:
            reading = self.look(target_id)

            # Tag not steadily in view: sweep one step and look again
            if reading is None:
                self.logger.log(
                    phase="search",
                    event="tag not seen -> sweep",
                    seen=False,
                    target_id=target_id,
                    commanded_deg=self.SEARCH_STEP_DEG,
                    **self._look_stats,
                    **self._enc(),
                )
                self.turn_in_place(+1, self.SEARCH_STEP_DEG, target_id=target_id)
                continue

            bearing_deg, distance_m = reading

            # Off-center on a trustworthy reading: turn once by that bearing, then confirm again
            if abs(bearing_deg) > self.CENTER_TOLERANCE_DEG:
                self.logger.log(
                    phase="center",
                    event="turn to center",
                    seen=True,
                    target_id=target_id,
                    bearing_raw=bearing_deg,
                    distance=distance_m,
                    Z=distance_m,
                    commanded_deg=bearing_deg,
                    **self._look_stats,
                    **self._enc(),
                )
                self.turn_in_place(np.sign(bearing_deg), abs(bearing_deg), target_id=target_id)
                continue

            # Centered and confirmed - include lidar as an independent distance cross-check
            self.logger.log(
                phase="center",
                event="centered",
                seen=True,
                target_id=target_id,
                bearing_raw=bearing_deg,
                distance=distance_m,
                Z=distance_m,
                **self._look_stats,
                **self._enc(),
                **self._lidar(),
            )
            return distance_m

    def drive_straight(self, distance_m, target_id=None):
        # Play back a trapezoidal speed curve over time to cover distance_m, both wheels equal (straight)
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

        # Log the planned (open-loop) move so we can compare commanded distance vs where the tag lands
        self.logger.log(
            phase="drive",
            event=f"trapezoid {distance_m:.2f}m / {t_total:.1f}s",
            target_id=target_id,
            distance=distance_m,
            **self._enc(),
        )

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

    def reach_tag(self, target_id):
        # Center, drive, then VERIFY we arrived. We drive open-loop (no encoders), so one trapezoid can
        # under/overshoot - take fresh captures afterward and re-drive the re-measured gap if we fell short.
        distance = self.approach_tag(target_id)  # initial search + center + confirmed distance
        for attempt in range(self.MAX_APPROACH_TRIES):
            remaining = distance - self.stop_tolerance_m
            if remaining <= self.GOAL_TOLERANCE_M:
                self.logger.log(
                    phase="verify",
                    event="reached",
                    seen=True,
                    target_id=target_id,
                    distance=distance,
                    Z=distance,
                    **self._enc(),
                    **self._lidar(),
                )
                return True  # confirmed within reach

            # Drive the measured gap, then look again to check we actually got there
            self.drive_straight(remaining, target_id)
            reading = self.look(target_id)

            # Tag no longer in view after driving its distance -> we're at (or past) it
            if reading is None:
                self.logger.log(
                    phase="verify",
                    event="tag lost after drive -> assume reached",
                    seen=False,
                    target_id=target_id,
                    **self._enc(),
                )
                return True

            # Still see it: update the distance, and re-aim if the blind drive left us off-center
            bearing_deg, distance = reading
            self.logger.log(
                phase="verify",
                event=f"re-measured (attempt {attempt + 1})",
                seen=True,
                target_id=target_id,
                bearing_raw=bearing_deg,
                distance=distance,
                Z=distance,
                **self._enc(),
            )
            if abs(bearing_deg) > self.CENTER_TOLERANCE_DEG:
                self.logger.log(
                    phase="center",
                    event="re-aim",
                    seen=True,
                    target_id=target_id,
                    bearing_raw=bearing_deg,
                    commanded_deg=bearing_deg,
                    **self._enc(),
                )
                self.turn_in_place(np.sign(bearing_deg), abs(bearing_deg), target_id=target_id)

        self.logger.log(
            phase="verify",
            event="gave up",
            seen=True,
            target_id=target_id,
            distance=distance,
            Z=distance,
            **self._enc(),
        )
        return False  # ran out of tries without confirming

    def update(self):
        print(
            f"Trapezoidal maze run over tags {self.MARKER_ID_LIST} - press Q in the window or Ctrl-C to quit"
        )

        try:
            # Visit each tag in order: center, drive the planned curve, then verify we arrived
            for target_id in self.MARKER_ID_LIST:
                print(f"Looking for tag {target_id}...")
                if self.reach_tag(target_id):
                    print(f"Reached tag {target_id} (verified)!")
                else:
                    print(
                        f"Could not confirm tag {target_id} after {self.MAX_APPROACH_TRIES} tries - moving on."
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


# Create our rover class and start the trapezoidal maze run
if __name__ == "__main__":
    rover = ArucoTrapezoidRunner()
    rover.update()
