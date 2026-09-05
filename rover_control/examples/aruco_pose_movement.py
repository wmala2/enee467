# External Libraries
import time

import cv2

# Local Libraries to Import
from ArUco_detector.aruco_pose_estimator import ArucoPoseEstimator

# Local Files to Import
from rover_control import conversions
from rover_control.pid import PID
from rover_control.rover import Rover


class ArucoFollower(Rover):
    MARKER_ID = 1  # the ArUco tag ID we drive toward

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

    def compute_wheel_speeds(self, pose):
        # Measure the time since the last control step for the PID math
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # The error is how much farther than the stop tolerance the tag is (+Z: forward)
        distance_error = pose[2] - self.stop_tolerance_m

        # Once we're inside the tolerance, stop and clear the controllers
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]

        # Forward speed comes from the distance PID, turning comes from the sideways offset PID
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(pose[0], dt)

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

        # Clamp the speed so it doesn't go insane
        speed = self.clamp(speed, min_pos, max_pos)
        return speed

    def update(self):
        print(
            f"Driving to ArUco tag {self.MARKER_ID}, stopping {self.stop_tolerance_m} m away - "
            f"press Q in the window to quit"
        )

        # Running Constantly
        while True:
            # Get the pose from our estimator
            frame, poses = self.estimator.process()

            if self.MARKER_ID in poses:
                print("Pose : ", poses[self.MARKER_ID]["position"])
                wheel_speeds = self.compute_wheel_speeds(poses[self.MARKER_ID]["position"])
            else:
                # No tag in sight, so hold still
                wheel_speeds = [0.0, 0.0]

            # Optional display if verbose=True
            if frame is not None:
                cv2.imshow("Aruco", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            self.write_real_velocities(wheel_speeds)
            # Sleep only the leftover time so commands go out at the firmware's rate
            self.sleep_to_command_rate()

        # Make sure the rover doesn't keep rolling after we quit
        self.stop()
        cv2.destroyAllWindows()


# Create our rover class and start following the tag
if __name__ == "__main__":
    rover = ArucoFollower()
    rover.update()
