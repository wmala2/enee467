# External Libraries
import time
from typing import ClassVar

import cv2
import numpy as np

# Local Libraries to Import
from ArUco_detector.aruco_pose_estimator import ArucoPoseEstimator

# Local Files to Import
from rover_control import conversions
from rover_control.encoder_poller import EncoderPoller
from rover_control.pid import PID
from rover_control.rover import Rover
from rover_control.run_logger import RunLogger


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

    # Cap forward speed well below the base class limit: at full throttle the chassis vibrates
    # enough to lose the ArUco tag every 2-5 frames. The logs show reliable lock only when
    # saturated=False (distance_error < ~0.3 m). 0.12 m/s keeps us in that regime the whole
    # approach.
    MAX_VELOCITY = 0.12  # m/s (overrides Rover.MAX_VELOCITY = 0.25)

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

        # PID on the forward distance to the tag (+Z is forward), output is forward speed in m/s
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)

        # PID on the sideways offset of the tag (+X is right), output is a turning speed in m/s
        self.heading_pid = PID(kp=0.6, ki=0.0, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # Remember when we last ran the controller so the PIDs get a real time step
        self._last_pid_time = time.perf_counter()

        # Diagnostic logging (one CSV row per tick/decision) + per-tick scratch from
        # compute_wheel_speeds
        self.logger = RunLogger("maze_pid", enabled=self.LOG)
        self._last_frame_stamp = 0.0
        self._dbg = {}

        # Background encoder + lidar poller - gives us actual wheel counts and proximity readings
        self.poller = EncoderPoller().start()

        # Populated by look() so center() can log detection reliability alongside its decisions
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
            cv2.imshow("Aruco", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def turn_in_place(self, direction, degrees, target_id=None):
        # Spin the rover on the spot by a number of degrees (open-loop, timed at the stall-floor
        # speed)
        # direction: +1 turns toward the tag's right (+X), -1 turns left
        omega = 2.0 * self.MIN_VELOCITY / self.wheel_separation
        spin_time = np.radians(degrees * self.TURN_SCALE) / omega
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
        # Stop-and-stare measurement, hardened: trust a reading only when several fresh frames
        # AGREE.
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

    def center(self, target_id):
        # Turn-look-turn until the tag is centered on a CONFIRMED reading.
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
            return

    def wheel_speeds_for(self, position):
        # Measure the time since the last control step for the PID math
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # The error is how much farther than the stop tolerance the tag is (+Z: forward)
        distance_error = position[2] - self.stop_tolerance_m
        if distance_error <= 0:
            self._dbg = {
                "distance_error": distance_error,
                "heading_input": position[0],
                "pid_forward": 0.0,
                "pid_turn": 0.0,
                "saturated": False,
            }
            return [0.0, 0.0]

        # Forward speed comes from the distance PID, turning comes from the sideways offset (X)
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(position[0], dt)

        # Mix forward and turn for a differential drive (tag to the right -> left wheel speeds up)
        speed = [forward + turn, forward - turn]

        # Correctly convert for our pseudo-twist message
        max_pos = conversions.convert_linear_vel_to_angular_vel(
            self.MAX_VELOCITY, self.wheel_diameter / 2.0
        )
        min_pos = conversions.convert_linear_vel_to_angular_vel(
            self.MIN_VELOCITY, self.wheel_diameter / 2.0
        )
        speed[0] = conversions.convert_linear_vel_to_angular_vel(
            speed[0], self.wheel_diameter / 2.0
        )
        speed[1] = conversions.convert_linear_vel_to_angular_vel(
            speed[1], self.wheel_diameter / 2.0
        )

        # Clamp the speed so it doesn't go insane (record whether the clamp actually fired)
        before = list(speed)
        speed = self.clamp(speed, min_pos, max_pos)
        self._dbg = {
            "distance_error": distance_error,
            "heading_input": position[0],
            "pid_forward": forward,
            "pid_turn": turn,
            "saturated": speed != before,
        }
        return speed

    def drive_to_tag(self, tag_id, stop_distance_m=None):
        # Continuous PID approach toward an already-centered tag. Returns "arrived" once we're
        # within the stop tolerance, or "recenter" if the tag drifts off-center or drops out of
        # view.
        #
        # stop_distance_m keeps this compatible with Rover.drive_to_tag, whose signature the
        # inherited run_maze() calls with that keyword. Without it, run_maze() raised TypeError
        # on this subclass. None means "use the tolerance this runner was constructed with".
        if stop_distance_m is not None:
            self.stop_tolerance_m = stop_distance_m

        # Start the controllers clean so the first dt isn't a stale stop-and-stare gap
        self.distance_pid.reset()
        self.heading_pid.reset()
        self._last_pid_time = time.perf_counter()
        _last_known_distance = None  # used to detect "tag went below FOV when very close"

        while True:
            # Pull the freshest frame + its capture time, detect (raw), then smooth (what we steer
            # on)
            frame, stamp = self.estimator.camera.latest()
            annotated, poses = self.estimator.detect(frame)
            self._show(annotated if annotated is not None else frame)

            seen = tag_id in poses
            raw = poses[tag_id] if seen else None
            bearing_raw = (
                np.degrees(np.arctan2(raw["position"][0], raw["position"][2])) if seen else None
            )

            # Smooth the pose in place - this EMA pose is what the heading PID actually acts on
            self.estimator._smooth(poses)
            position = poses[tag_id]["position"] if seen else None
            bearing_smoothed = np.degrees(np.arctan2(position[0], position[2])) if seen else None

            # How old is this frame, and is it actually new since last tick? (camera-lag
            # diagnostics)
            frame_age = (time.perf_counter() - stamp) if stamp else None
            frame_is_new = stamp != self._last_frame_stamp
            self._last_frame_stamp = stamp

            # Lost the tag mid-approach: if we were already close, the tag likely slipped below the
            # camera's downward FOV - declare arrived. Otherwise back out and re-center.
            if not seen:
                if (
                    _last_known_distance is not None
                    and _last_known_distance < self.stop_tolerance_m + 0.15
                ):
                    self.logger.log(
                        phase="drive",
                        event="tag lost near goal -> assumed arrived",
                        seen=False,
                        tag_id=tag_id,
                        distance=_last_known_distance,
                        frame_stamp=stamp,
                        frame_age=frame_age,
                        frame_is_new=frame_is_new,
                        **self._enc(),
                        **self._lidar(),
                    )
                    self.stop()
                    return "arrived"
                self.logger.log(
                    phase="drive",
                    event="lost tag",
                    seen=False,
                    tag_id=tag_id,
                    frame_stamp=stamp,
                    frame_age=frame_age,
                    frame_is_new=frame_is_new,
                    **self._enc(),
                )
                self.stop()
                return "recenter"

            # Arrived: within the forward stop tolerance
            if position[2] - self.stop_tolerance_m <= 0:
                self.logger.log(
                    phase="drive",
                    event="arrived",
                    seen=True,
                    tag_id=tag_id,
                    Z=position[2],
                    distance=raw["distance"],
                    frame_stamp=stamp,
                    frame_age=frame_age,
                    frame_is_new=frame_is_new,
                    **self._enc(),
                    **self._lidar(),
                )
                self.stop()
                return "arrived"

            # Drifted too far off-center to trust a straight approach: stop and re-center
            if abs(bearing_smoothed) > self.DRIVE_RECENTER_DEG:
                self.logger.log(
                    phase="drive",
                    event=f"recenter (bearing {bearing_smoothed:.1f})",
                    seen=True,
                    tag_id=tag_id,
                    bearing_raw=bearing_raw,
                    bearing_smoothed=bearing_smoothed,
                    Z=position[2],
                    frame_stamp=stamp,
                    frame_age=frame_age,
                    frame_is_new=frame_is_new,
                    **self._enc(),
                )
                self.stop()
                return "recenter"

            _last_known_distance = position[2]

            # Otherwise keep driving toward it with the PID, logging the full tick
            wheel_speeds = self.wheel_speeds_for(position)
            self.logger.log(
                phase="drive",
                seen=True,
                tag_id=tag_id,
                bearing_raw=bearing_raw,
                bearing_smoothed=bearing_smoothed,
                X=position[0],
                Y=position[1],
                Z=position[2],
                distance=raw["distance"],
                reproj_error=raw.get("reproj_error"),
                frame_stamp=stamp,
                frame_age=frame_age,
                frame_is_new=frame_is_new,
                distance_error=self._dbg.get("distance_error"),
                heading_input=self._dbg.get("heading_input"),
                pid_forward=self._dbg.get("pid_forward"),
                pid_turn=self._dbg.get("pid_turn"),
                cmd_left=wheel_speeds[0],
                cmd_right=wheel_speeds[1],
                saturated=self._dbg.get("saturated"),
                **self._enc(),
            )
            self.write_real_velocities(wheel_speeds)
            self.sleep_to_command_rate()

    def update(self):
        print(
            f"Driving ArUco tags {self.MARKER_ID_LIST}, stopping {self.stop_tolerance_m} m away - "
            f"press Q in the window or Ctrl-C to quit"
        )

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


# Create our rover class and start running the maze
if __name__ == "__main__":
    rover = ArucoRunner()
    rover.update()
