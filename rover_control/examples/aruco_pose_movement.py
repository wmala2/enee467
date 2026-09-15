"""Drives the rover straight up to one ArUco tag and stops a fixed distance away.

We read the tag's (X, Y, Z) position from the camera every loop, feed that into two small
feedback controllers (PIDs) -- one to control forward/backward speed, one to control steering --
and turn their outputs into left/right wheel speeds.
"""

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
    """A rover that finds one ArUco tag and drives up to it.

    `Rover` already knows how to talk to the wheels over the network, so we only need to add
    the "find the tag, then steer toward it" logic on top.
    """

    # Every ArUco tag printed out has a number on it. This is the one we'll look for.
    MARKER_ID = 1

    def __init__(self, stop_tolerance_m=0.25):
        # Tell the base Rover class which network address to send wheel commands to.
        super().__init__(rover_addr="192.168.50.214")

        # How close (in meters) the rover should get to the tag before it stops driving.
        self.stop_tolerance_m = stop_tolerance_m

        # Point the rover at the camera's video stream so it can actually see the tag.
        img_addr = "http://192.168.50.114:80/capture"
        self.estimator = ArucoPoseEstimator(
            http_addr=img_addr,
            verbose=True,  # set False if you only want data, no preview window
        )

        # A PID controller is a small feedback loop: measure how wrong you are (the error), and
        # nudge your output based on how big that error is and how fast it's changing. We need
        # one PID to control forward speed and a second one to control steering.
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)
        self.heading_pid = PID(kp=0.6, ki=0.0, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # PID math needs to know how much time passed since the last update, so remember it here.
        self._last_pid_time = time.perf_counter()

    def wheel_speeds_for(self, pose):
        """Turns one tag position reading into a pair of [left, right] wheel speeds."""
        # Every control loop needs a timestep (dt): how much time passed since we last checked.
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # The "error" here is simple: how much farther away is the tag than where we want to stop?
        distance_error = pose[2] - self.stop_tolerance_m

        # Already close enough? Stop, and reset the PIDs so old error doesn't carry into next time.
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]

        # Feed each error into its own PID to get a forward speed and a turning speed.
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(pose[0], dt)

        # A differential-drive rover turns by spinning its two wheels at different speeds: add
        # the turn amount to one side and subtract it from the other (tag to the right -> left
        # wheel speeds up).
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
            f"Driving to ArUco tag {self.MARKER_ID}, stopping {self.stop_tolerance_m} m away - "
            f"press Q in the window to quit"
        )

        while True:
            # Ask the camera for its newest picture and any ArUco tag positions found in it.
            frame, poses = self.estimator.process()

            # If we can see our target tag, steer toward it. If not, sit still instead of
            # driving blind.
            if self.MARKER_ID in poses:
                print("Pose : ", poses[self.MARKER_ID]["position"])
                wheel_speeds = self.wheel_speeds_for(poses[self.MARKER_ID]["position"])
            else:
                wheel_speeds = [0.0, 0.0]

            # Optional preview window (only pops up if verbose=True above); press Q to quit.
            if frame is not None:
                cv2.imshow("Aruco", frame)
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
    rover = ArucoFollower()
    rover.update()
