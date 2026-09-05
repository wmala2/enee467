# External Libraries
import json
import time

import numpy as np

from rover_control import conversions

# Local Files to Import
from rover_control import network_interface


class Rover:
    COMMAND_RATE_HZ = 10.0  # the rover firmware listens for commands at about this rate
    MAX_VELOCITY = 0.25  # m/s
    MIN_VELOCITY = 0.075  # m/s
    # Open-loop turns are timed, not encoder-measured, so they can under- or overshoot.
    # If turns consistently land short, increase this (e.g. 1.5 if turns are ~30% short).
    TURN_SCALE = 1.0

    # --- Settings for the simple one-line behaviors (drive_to_tag, run_maze, ...) ---
    DEFAULT_CAMERA_ADDR = "http://192.168.50.123:80/capture"
    CRUISE_VELOCITY = 0.20  # top speed of the trapezoidal drive (m/s)
    ACCEL = 0.20  # how fast forward() ramps up and down (m/s^2)
    CENTER_TOLERANCE_DEG = 8.0  # "centered enough" on a tag to drive straight at it
    SEARCH_STEP_DEG = 30.0  # how far we turn to sweep for a tag we can't see
    SETTLE_S = 0.3  # pause after a turn so the chassis stops before we look
    CONFIRM_FRAMES = 5  # fresh frames (captured after we stopped) sampled per look
    CONFIRM_MIN = 3  # tag must appear in at least this many fresh frames to trust it
    BEARING_AGREE_DEG = 4.0  # those fresh frames must agree within this bearing spread
    MAX_LOOK_TRIES = 3  # sampling rounds before we treat the tag as unseen

    def __init__(
        self,
        wheel_diameter_m=(70.0 / 1000.0),
        wheel_separation_m=(187.832 / 1000.0),
        camera_addr=DEFAULT_CAMERA_ADDR,
        show_camera=False,
    ):
        # Physical geometry of the rover, used for all of the speed conversions
        self.wheel_diameter = wheel_diameter_m
        self.wheel_separation = wheel_separation_m
        self.velocity_bias = (self.MAX_VELOCITY + self.MIN_VELOCITY) / 2.0
        self._last_loop_time = time.perf_counter()

        # Camera for the simple vision behaviors (estimators are created the first time they're needed)
        self.camera_addr = camera_addr
        self.show_camera = show_camera
        self._aruco_estimator = None
        self._yolo_estimator = None

        # Incrementing index the firmware can use to order/deduplicate our velocity commands
        self._cmd_index = 0

    def compute_wheel_speeds(self):
        # Subclasses decide HOW the rover moves by returning [left, right] wheel speeds in rad/s
        raise NotImplementedError("Extend Rover and implement compute_wheel_speeds()")

    def update(self):
        # Out-of-the-box control loop: compute speeds, send them, repeat at the command rate
        while True:
            wheel_speeds = self.compute_wheel_speeds()
            self.write_real_velocities(wheel_speeds)
            self.sleep_to_command_rate()

    def write_real_velocities(self, wheel_velocities):
        # Convert our values to the correct linear velocity
        left_m = conversions.convert_angular_vel_to_linear_vel(
            wheel_velocities[0], self.wheel_diameter / 2.0
        )
        right_m = conversions.convert_angular_vel_to_linear_vel(
            wheel_velocities[1], self.wheel_diameter / 2.0
        )

        # Put message into JSON format for the rover firmware.
        # Keys MUST match the firmware exactly: command "m" with left_mps/right_mps (meters per second).
        self._cmd_index += 1
        msg = json.dumps({
            "command": "m",
            "left_mps": left_m,
            "right_mps": right_m,
            "index": self._cmd_index,
        }).encode("utf-8")
        print(f"V: {left_m:.3f} {right_m:.3f}")
        network_interface.send_message(msg)

    def stop(self):
        # Send zero velocity so the rover halts right away
        self.write_real_velocities([0.0, 0.0])

    def sleep_to_command_rate(self):
        # Sleep only the leftover loop time so commands actually go out at COMMAND_RATE_HZ
        elapsed = time.perf_counter() - self._last_loop_time
        remaining = (1.0 / self.COMMAND_RATE_HZ) - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_loop_time = time.perf_counter()

    def clamp(self, speed, min_pos, max_pos):
        # Keep wheel speeds inside the rover's physical limits (motors stall below the minimum)
        for i in range(2):
            if speed[i] < -max_pos:
                speed[i] = -max_pos
            elif speed[i] < 0 and speed[i] > -min_pos:
                speed[i] = -min_pos
            elif speed[i] > 0 and speed[i] < min_pos:
                speed[i] = min_pos
            elif speed[i] > max_pos:
                speed[i] = max_pos
        return speed

    # =====================================================================================
    # Simple one-line API - students call these directly, the control details stay hidden
    # =====================================================================================

    def _aruco(self):
        # Create the ArUco estimator + background camera the first time a vision method runs
        if self._aruco_estimator is None:
            from ArUco_detector.aruco_pose_estimator import ArucoPoseEstimator

            self._aruco_estimator = ArucoPoseEstimator(
                http_addr=self.camera_addr, verbose=self.show_camera
            )
        return self._aruco_estimator

    def _maybe_show(self, frame):
        # Pop up the camera view only if the student asked for it
        if self.show_camera and frame is not None:
            import cv2

            cv2.imshow("Rover", frame)
            cv2.waitKey(1)

    def turn(self, degrees):
        # Spin in place by `degrees` (positive = turn right / toward +X), then stop and settle
        spin = conversions.convert_linear_vel_to_angular_vel(
            self.MIN_VELOCITY, self.wheel_diameter / 2.0
        )
        left, right = (spin, -spin) if degrees >= 0 else (-spin, spin)

        # Spinning at the stall-floor speed, the rover turns at this rate, so this long covers `degrees`
        omega = 2.0 * self.MIN_VELOCITY / self.wheel_separation
        end_time = time.perf_counter() + np.radians(abs(degrees) * self.TURN_SCALE) / omega
        try:
            while time.perf_counter() < end_time:
                self.write_real_velocities([left, right])
                self.sleep_to_command_rate()
        finally:
            # Always halt - even on Ctrl-C - so the rover can never run away mid-turn
            self.stop()
        time.sleep(self.SETTLE_S)

    def forward(self, distance_m):
        # Drive straight `distance_m` meters with a trapezoidal speed curve (negative = backward)
        direction = 1.0 if distance_m >= 0 else -1.0
        distance = abs(distance_m)
        v_min, v_cruise, a = self.MIN_VELOCITY, self.CRUISE_VELOCITY, self.ACCEL

        # One ramp goes from the stall floor up to cruise; work out its time and distance
        t_ramp = (v_cruise - v_min) / a
        d_ramp = (v_min + v_cruise) / 2.0 * t_ramp

        # Too short to reach cruise -> use a triangle (lower peak speed)
        if 2.0 * d_ramp >= distance:
            v_top = min(v_cruise, np.sqrt(v_min**2 + a * distance))
            t_ramp = (v_top - v_min) / a
            t_cruise = 0.0
        else:
            v_top = v_cruise
            t_cruise = (distance - 2.0 * d_ramp) / v_cruise
        t_total = 2.0 * t_ramp + t_cruise

        # Play back the speed curve at the command rate
        start = time.perf_counter()
        try:
            while True:
                t = time.perf_counter() - start
                if t >= t_total:
                    break
                if t < t_ramp:
                    v = v_min + a * t
                elif t < t_ramp + t_cruise:
                    v = v_top
                else:
                    v = v_top - a * (t - t_ramp - t_cruise)
                v = max(v, v_min) * direction  # never below the stall floor, then apply direction
                wheel = conversions.convert_linear_vel_to_angular_vel(v, self.wheel_diameter / 2.0)
                self.write_real_velocities([wheel, wheel])
                self.sleep_to_command_rate()
        finally:
            # Always halt - even on Ctrl-C - so the rover can never run away mid-drive
            self.stop()

    def see_tag(self, tag_id):
        # One stop-and-stare reading of an ArUco tag. Returns {"position":[x,y,z], "bearing":deg,
        # "distance":m} averaged over fresh, agreeing frames, or None if the tag isn't steadily seen.
        estimator = self._aruco()
        for _ in range(self.MAX_LOOK_TRIES):
            since = (
                time.perf_counter()
            )  # only trust frames captured after right now (beats camera lag)
            positions, bearings = [], []
            for _ in range(self.CONFIRM_FRAMES):
                frame, since = estimator.next_frame(since)
                if frame is None:
                    break
                _, poses = estimator.detect(frame)
                self._maybe_show(frame)
                if tag_id in poses:
                    position = poses[tag_id]["position"]
                    positions.append(position)
                    bearings.append(np.degrees(np.arctan2(position[0], position[2])))

            # Trust it only if enough fresh frames saw the tag and they agree on the bearing
            if (
                len(bearings) >= self.CONFIRM_MIN
                and (max(bearings) - min(bearings)) <= self.BEARING_AGREE_DEG
            ):
                average = np.mean(np.array(positions), axis=0).tolist()
                return {
                    "position": average,
                    "bearing": float(np.mean(bearings)),
                    "distance": float(np.linalg.norm(average)),
                }
        return None

    def _center_on_tag(self, tag_id):
        # Turn-look-turn until the tag is centered on a confirmed reading; returns its distance (m)
        while True:
            reading = self.see_tag(tag_id)
            if reading is None:
                self.turn(self.SEARCH_STEP_DEG)  # can't see it: sweep one step and look again
                continue
            if abs(reading["bearing"]) > self.CENTER_TOLERANCE_DEG:
                self.turn(reading["bearing"])  # turn by the measured bearing to face it
                continue
            return reading["distance"]

    def drive_to_tag(self, tag_id, stop_distance_m=0.25):
        # Search for an ArUco tag, center on it, and drive up to `stop_distance_m` away
        print(f"Driving to tag {tag_id}...")
        distance = self._center_on_tag(tag_id)
        self.forward(max(distance - stop_distance_m, 0.0))
        print(f"Reached tag {tag_id}.")

    def run_maze(self, tag_ids, stop_distance_m=0.25):
        # Drive a list of ArUco tags in order
        for tag_id in tag_ids:
            self.drive_to_tag(tag_id, stop_distance_m=stop_distance_m)
        print("Maze complete - every tag visited!")
        self.stop()

    def _yolo(self):
        # Create the YOLO 3D-pose estimator the first time an object-following method runs
        if self._yolo_estimator is None:
            from pathlib import Path

            from YOLO_agent.yolo_pose_estimator import YOLOPoseEstimator

            model_path = (
                Path(__file__).resolve().parents[1] / "YOLO_agent" / "models" / "yolov8n.pt"
            )
            # imgsz=960 sees smaller/farther objects; the depth pass makes this a slow (~1 Hz) look
            self._yolo_estimator = YOLOPoseEstimator(
                model_path=model_path, imgsz=640, verbose=self.show_camera
            )
        return self._yolo_estimator

    def _see_object(self, yolo, name):
        # Stop-and-stare reading of a named COCO object: returns {"bearing":deg, "distance":m} or None.
        # Aims at the nearest matching object and only trusts it if several frames agree.
        bearings, distances = [], []
        for _ in range(self.CONFIRM_FRAMES):
            try:
                frame = yolo.get_frame_from_http(self.camera_addr)
            except Exception:
                continue
            annotated, detections = yolo.process(frame)
            self._maybe_show(annotated)
            matches = [d for d in detections if d["name"] == name]
            if matches:
                nearest = min(matches, key=lambda d: d["position"][2])  # smallest Z = closest
                x, z = nearest["position"][0], nearest["position"][2]
                bearings.append(np.degrees(np.arctan2(x, z)))
                distances.append(z)

        if (
            len(bearings) >= self.CONFIRM_MIN
            and (max(bearings) - min(bearings)) <= self.BEARING_AGREE_DEG
        ):
            return {"bearing": float(np.mean(bearings)), "distance": float(np.mean(distances))}
        return None

    def follow_object(self, name, stop_distance_m=0.25):
        # Search for a named COCO object (e.g. "bottle"), center on it, and drive up to stop_distance_m away
        print(f"Looking for a {name}...")
        yolo = self._yolo()
        while True:
            reading = self._see_object(yolo, name)
            if reading is None:
                self.turn(self.SEARCH_STEP_DEG)  # can't see it: sweep one step and look again
                continue
            if abs(reading["bearing"]) > self.CENTER_TOLERANCE_DEG:
                self.turn(reading["bearing"])  # turn by the measured bearing to face it
                continue
            self.forward(max(reading["distance"] - stop_distance_m, 0.0))
            print(f"Reached the {name}.")
            return

    def run_object_maze(self, names, stop_distance_m=0.25):
        # Drive to a list of COCO objects in order (the YOLO version of run_maze)
        for name in names:
            self.follow_object(name, stop_distance_m=stop_distance_m)
        print("Object maze complete - every object visited!")
        self.stop()

    def close(self):
        # Shut down the background camera reader (safe to skip - it also stops when the program ends)
        if self._aruco_estimator is not None:
            self._aruco_estimator.close()
