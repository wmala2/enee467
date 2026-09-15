"""Drives the rover up to one named object (default: a bottle) and stops a fixed distance away.

This is the same idea as aruco_pose_movement.py, but instead of looking for a printed ArUco tag,
it uses a YOLO object detector to find an everyday object and estimate its (X, Y, Z) position.
"""

# External Libraries
from pathlib import Path
import time

import cv2

# Local Files to Import
from rover_control import conversions
from rover_control.pid import PID
from rover_control.rover import Rover

# Local Libraries to Import
from YOLO_agent.yolo_pose_estimator import YOLOPoseEstimator


class YoloFollower(Rover):
    """A rover that finds one named COCO object and drives up to it."""

    # Any label YOLO knows (see COCO's class list) works here, not just "bottle".
    TARGET_OBJECT = "bottle"

    def __init__(self, stop_tolerance_m=0.25):
        super().__init__()

        # How close (in meters) the rover should get to the object before it stops driving.
        self.stop_tolerance_m = stop_tolerance_m

        # Point the rover at the camera's video stream so it can actually see the object.
        self.img_addr = "http://192.168.50.123:80/capture"

        # YOLOPoseEstimator needs its trained weights file; this locates the copy that ships
        # with the YOLO_agent folder regardless of where this script is run from.
        model_path = Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"

        # Create our YOLO 3D pose estimator (imgsz=960 finds smaller/farther objects than the
        # detector's default resolution would).
        self.estimator = YOLOPoseEstimator(
            model_path=model_path,
            imgsz=960,
            verbose=True,  # set False if you only want data, no preview window
        )

        # A PID controller is a small feedback loop: measure how wrong you are (the error), and
        # nudge your output based on how big that error is and how fast it's changing. We need
        # one PID to control forward speed and a second one to control steering.
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)
        self.heading_pid = PID(kp=0.6, ki=0.0, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # PID math needs to know how much time passed since the last update, so remember it here.
        self._last_pid_time = time.perf_counter()

    def wheel_speeds_for(self, position):
        """Turns one object position reading into a pair of [left, right] wheel speeds."""
        # Every control loop needs a timestep (dt): how much time passed since we last checked.
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # The "error" here is simple: how much farther away is the object than where we want to
        # stop?
        distance_error = position[2] - self.stop_tolerance_m

        # Already close enough? Stop, and reset the PIDs so old error doesn't carry into next time.
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]

        # Feed each error into its own PID to get a forward speed and a turning speed.
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(position[0], dt)

        # A differential-drive rover turns by spinning its two wheels at different speeds: add
        # the turn amount to one side and subtract it from the other (object to the right ->
        # left wheel speeds up).
        speed = [forward + turn, forward - turn]

        # The rover's firmware expects wheel speed in radians/second, not meters/second, so
        # convert both our speeds -- and the min/max limits we'll clamp against -- to match.
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

        # Never send a speed faster than the rover can safely handle.
        speed = self.clamp(speed, min_pos, max_pos)
        return speed

    def update(self):
        """The main loop: look, decide how fast to drive, send the command, repeat."""
        print(
            f"Driving to YOLO object '{self.TARGET_OBJECT}', stopping {self.stop_tolerance_m} m "
            f"away - press Q in the window to quit"
        )

        while True:
            # Grab one JPEG still from the ESP32 camera, then find the object and its 3D pose.
            # A single bad frame (dropped packet, camera hiccup) shouldn't crash a moving rover,
            # so we log it and just try again on the next loop instead of stopping.
            try:
                frame = self.estimator.get_frame_from_http(self.img_addr)
            except Exception as error:  # noqa: BLE001 -- one bad frame should not stop the rover mid-drive
                print(f"Camera error: {error}")
                continue
            annotated, detections = self.estimator.process(frame)

            # Keep only detections matching our target object, and aim at the nearest one if
            # several show up in frame at once.
            targets = [d for d in detections if d["name"] == self.TARGET_OBJECT]
            if targets:
                nearest = min(targets, key=lambda d: d["position"][2])
                print("Pose : ", nearest["position"])
                wheel_speeds = self.wheel_speeds_for(nearest["position"])
            else:
                # Object not in sight, so hold still instead of driving blind.
                wheel_speeds = [0.0, 0.0]

            # Optional preview window (only pops up if verbose=True above); press Q to quit.
            if annotated is not None:
                cv2.imshow("YOLO Follower", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Actually send the command, then wait just long enough to hit our fixed control rate.
            self.write_real_velocities(wheel_speeds)
            self.sleep_to_command_rate()

        # Always stop the motors and close windows when we're done -- never leave a rover rolling.
        self.stop()
        cv2.destroyAllWindows()


# Everything above just defines how the rover should behave. This actually creates one and
# tells it to go.
if __name__ == "__main__":
    rover = YoloFollower()
    rover.update()
