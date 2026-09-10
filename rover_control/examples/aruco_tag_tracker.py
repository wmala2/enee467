# External Libraries
import time

import cv2

# Local Libraries to Import
from ArUco_detector.aruco_pose_estimator import ArucoPoseEstimator

# Local Files to Import
from rover_control import conversions
from rover_control.pid import PID
from rover_control.rover import Rover


class ArucoTracker(Rover):
    MARKER_ID = 0  # the ArUco tag ID we keep tracking
    STANDOFF_M = 0.40  # the distance (m) the rover tries to hold from a moving tag
    DEADBAND_M = 0.05  # don't bother driving while we're within this much of the standoff
    SEARCH_PATIENCE_FRAMES = 5  # missed frames before we spin to look for the tag again

    def __init__(self):
        super().__init__()

        # Camera Stream Address
        img_addr = "http://192.168.50.123:80/capture"

        # Create our ArUco Marker Pose Estimator
        self.estimator = ArucoPoseEstimator(
            http_addr=img_addr,
            verbose=True,  # set False if you only want data
        )

        # PID that holds the standoff distance (+Z forward): drives forward when far, backs up when
        # close
        self.distance_pid = PID(kp=0.8, ki=0.05, kd=0.10, output_limit=self.MAX_VELOCITY)

        # PID that keeps the tag centered (+X is right), output is a turning speed in m/s
        self.heading_pid = PID(kp=0.7, ki=0.0, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # Remember when we last ran the controller so the PIDs get a real time step
        self._last_pid_time = time.perf_counter()

        # Count frames where the tag was missing so we know when to search
        self.frames_without_tag = 0

    def wheel_speeds_for(self, position):
        # Measure the time since the last control step for the PID math
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # Positive error means the tag is farther than our standoff, negative means too close
        distance_error = position[2] - self.STANDOFF_M

        # Inside the deadband we hold our ground on distance, but still turn to stay centered
        forward = 0.0
        if abs(distance_error) > self.DEADBAND_M:
            forward = self.distance_pid.update(distance_error, dt)

        # Turning keeps the tag in the middle of the frame as it moves left and right
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

        # Clamp the speed so it doesn't go insane
        speed = self.clamp(speed, min_pos, max_pos)
        return speed

    def search_speeds(self):
        # Spin slowly in place to bring a lost tag back into view (left wheel back, right wheel
        # forward)
        spin = conversions.convert_linear_vel_to_angular_vel(
            self.MIN_VELOCITY, self.wheel_diameter / 2.0
        )
        return [-spin, spin]

    def update(self):
        print(
            f"Tracking ArUco tag {self.MARKER_ID}, holding {self.STANDOFF_M} m - press Q in the "
            f"window to quit"
        )

        # Running Constantly
        while True:
            # Get the pose from our estimator
            frame, poses = self.estimator.process()

            if self.MARKER_ID in poses:
                # Tag is in view: reset the lost-frame counter and chase the standoff distance
                self.frames_without_tag = 0
                print("Pose : ", poses[self.MARKER_ID]["position"])
                wheel_speeds = self.wheel_speeds_for(poses[self.MARKER_ID]["position"])
            else:
                # Tag missing: wait a few frames, then spin in place to find it again
                self.frames_without_tag += 1
                if self.frames_without_tag > self.SEARCH_PATIENCE_FRAMES:
                    wheel_speeds = self.search_speeds()
                else:
                    wheel_speeds = [0.0, 0.0]

            # Optional display if verbose=True
            if frame is not None:
                cv2.imshow("Aruco Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            self.write_real_velocities(wheel_speeds)
            # Sleep only the leftover time so commands go out at the firmware's rate
            self.sleep_to_command_rate()

        # Make sure the rover doesn't keep rolling after we quit
        self.stop()
        cv2.destroyAllWindows()


# Create our rover class and start tracking the tag
if __name__ == "__main__":
    rover = ArucoTracker()
    rover.update()
