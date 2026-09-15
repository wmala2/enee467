# ArUco Pose Movement

This example drives the rover straight at a single ArUco tag and stops a set distance away, using two PID controllers (one for forward distance, one for steering) that react to every camera frame.

## Prerequisites

- A rover with the ESP32 camera reachable at `http://192.168.50.123:80/capture` (the address is hardcoded in the script as `img_addr`)
- The rover's UDP control endpoint reachable at the default `192.168.50.223` (or set the `ROVER_IP` environment variable), since `rover.py` sends wheel commands over `network_interface`
- `uv sync` already run from the repo root (this pulls in `opencv-python`, `numpy`, and the workspace's `ArUco_detector` package)
- An ArUco tag from the `DICT_4X4_250` dictionary, printed with a 10 cm black-border square (the `ArucoPoseEstimator` default `marker_length_m`), with **ID 1** — that's the `MARKER_ID` this script looks for
- A display available for `cv2.imshow`, since the script pops up a window and quits on `Q`

## Run it

```bash
uv run rover_control/examples/aruco_pose_movement.py
```

There are no command-line arguments — the script has no `argparse` or `sys.argv` handling, just a plain `if __name__ == "__main__":` block that constructs `ArucoFollower()` and calls `.update()`. Once running, a window pops up showing the camera feed, and the rover drives toward tag ID 1, slowing to a stop once it's within 0.25 m.

## How it works

The class extends the shared `Rover` base class and sets up two PID loops plus the pose estimator in its constructor.

```python
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
```

A PID controller is just a feedback loop that turns "how wrong am I right now" into "how hard should I push" — here one PID handles closing the gap to the tag, and a second, independent PID handles keeping the tag centered in view so the rover drives straight instead of curving in.

```python
        # PID on the forward distance to the tag (+Z is forward), output is forward speed in m/s
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)

        # PID on the sideways offset of the tag (+X is right), output is a turning speed in m/s
        self.heading_pid = PID(kp=0.6, ki=0.0, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # Remember when we last ran the controller so the PIDs get a real time step
        self._last_pid_time = time.perf_counter()
```

`wheel_speeds_for` is the per-frame decision function: given the tag's 3D position, it works out how fast each wheel should spin. It starts by measuring the real elapsed time (`dt`) since the last call, because PID math needs an actual time step, not an assumed one.

```python
    def wheel_speeds_for(self, pose):
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
```

Once it's confirmed the rover still needs to move, it asks each PID for its output — a forward speed and a turning speed — then mixes them into left/right wheel speeds the way a differential-drive robot always does: add the turn to one wheel and subtract it from the other.

```python
        # Forward speed comes from the distance PID, turning comes from the sideways offset PID
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(pose[0], dt)

        # Mix forward and turn for a differential drive (tag to the right -> left wheel speeds up)
        speed = [forward + turn, forward - turn]
```

The raw PID output is a linear speed in meters/second, but the rover firmware wants wheel angular velocity, and speeds also need to be kept within what the motors can physically do — too slow and they stall, too fast and it's unsafe.

```python
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
```

Finally, `update()` is the main loop: grab a frame and any detected tag poses from the estimator, decide wheel speeds (or hold still if the tag isn't visible), show the debug window, and send the command at the firmware's expected rate.

```python
    def update(self):
        # Running Constantly
        while True:
            # Get the pose from our estimator
            frame, poses = self.estimator.process()

            if self.MARKER_ID in poses:
                print("Pose : ", poses[self.MARKER_ID]["position"])
                wheel_speeds = self.wheel_speeds_for(poses[self.MARKER_ID]["position"])
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
```

## See also

- [rover_control/](../../rover_control/) — the Rover base class and other examples
- [ArUco_detector/](../../ArUco_detector/) — the marker detection module this example depends on
- [Back to README](../../README.md)
