# ArUco Tag Tracker

This example makes the rover follow a moving ArUco tag, holding a fixed standoff distance rather than stopping right in front of it — if you walk the tag around, the rover chases it, and if the tag disappears from view for too long, it spins in place to hunt for it again.

## Prerequisites

- A rover with the ESP32 camera reachable at `http://192.168.50.123:80/capture` (hardcoded as `img_addr` in the script)
- The rover's UDP control endpoint reachable at the default `192.168.50.223` (or set `ROVER_IP`), since wheel commands go out over `network_interface`
- `uv sync` already run from the repo root (installs `opencv-python`, `numpy`, and the workspace's `ArUco_detector` package)
- An ArUco tag from `DICT_4X4_250`, printed at the default 10 cm marker size, with **ID 0** — that's the `MARKER_ID` this script tracks
- Open floor space, since the rover will actively follow the tag as you move it
- A display for `cv2.imshow`, since the script opens a window and quits on `Q`

## Run it

```bash
uv run rover_control/examples/aruco_tag_tracker.py
```

The script takes no CLI arguments — it's a plain `if __name__ == "__main__":` block creating `ArucoTracker()` and calling `.update()`. Once running, a window shows the camera feed, and the rover holds itself about 0.40 m from tag ID 0, driving forward/backward and turning to keep that distance as the tag moves, and spinning in place to search if it loses the tag for more than a few frames.

## How it works

Like the other ArUco examples, this extends `Rover` and sets class constants for the behavior it wants: which tag to follow, how far to stay from it, and how patient to be before considering the tag "lost."

```python
class ArucoTracker(Rover):
    MARKER_ID = 0  # the ArUco tag ID we keep tracking
    STANDOFF_M = 0.40  # the distance (m) the rover tries to hold from a moving tag
    DEADBAND_M = 0.05  # don't bother driving while we're within this much of the standoff
    SEARCH_PATIENCE_FRAMES = 5  # missed frames before we spin to look for the tag again
```

The constructor wires up the camera pose estimator and two PID controllers — one for keeping distance, one for keeping the tag centered — the same distance/heading split used in the other ArUco examples.

```python
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
```

The key difference from `aruco_pose_movement.py` is the *deadband*: instead of always chasing an exact distance (which would make the rover twitch constantly), it only drives forward/backward once the error is bigger than a small tolerance, so tiny jitter in the tag's apparent distance doesn't turn into wheel motion.

```python
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
```

Just like the pose-movement example, the forward and turn speeds get mixed into left/right wheel commands, converted from linear to angular velocity, then clamped so the motors never get an impossible speed.

```python
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
```

When the tag disappears from the frame — say, someone carries it out of view — the rover doesn't want to sit frozen forever, so it has a dedicated "spin slowly and look around" command.

```python
    def search_speeds(self):
        # Spin slowly in place to bring a lost tag back into view (left wheel back, right wheel
        # forward)
        spin = conversions.convert_linear_vel_to_angular_vel(
            self.MIN_VELOCITY, self.wheel_diameter / 2.0
        )
        return [-spin, spin]
```

The main loop ties it together: while the tag is visible it resets the "missing" counter and chases the standoff distance; once the tag has been missing for more than `SEARCH_PATIENCE_FRAMES`, it switches to the search spin instead of just idling.

```python
    def update(self):
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
```

## See also

- [rover_control/](../../rover_control/) — the Rover base class and other examples
- [ArUco_detector/](../../ArUco_detector/) — the marker detection module this example depends on
- [Back to README](../../README.md)
