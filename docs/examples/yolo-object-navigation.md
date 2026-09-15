# YOLO Object Navigation

This example drives the physical rover toward a named object (a bottle, by default) using the YOLO 3D pose estimator to continuously measure how far away and how far off-center the object is, then steers with two PID controllers until it stops a set distance away. It's the "under the hood" version of the simple `rover.follow_object()` API call — useful for students who want to see or extend the actual control loop.

## Prerequisites

- The physical rover, its ESP32 camera reachable over HTTP (default `http://192.168.50.123:80/capture`), and its motors/firmware listening for velocity commands
- The `rover_control` package's dependencies (see `rover_control/README.md` — if you've run `pip install -e .` from the repo root, these are already installed)
- `ultralytics`, `torch`, `transformers`, and `Pillow` for the YOLO 3D pose estimator (same stack as the 3D pose example), plus the YOLOv8n weights at `YOLO_agent/models/yolov8n.pt`
- A target object from the COCO class list in view of the rover's camera (default target is `"bottle"`, configurable by editing `YoloFollower.TARGET_OBJECT`)
- Open floor space — the rover will actually drive during this example

## Run it

```bash
uv run rover_control/examples/yolo_object_movement.py
```

The rover repeatedly looks for the target object, turns and drives toward the nearest match, and stops once it's within the configured tolerance (0.25 m by default); an OpenCV window shows the camera view with detections overlaid, and pressing `q` in that window stops the rover and exits.

## How it works

The script defines `YoloFollower`, a subclass of the shared `Rover` base class (`rover_control/rover.py`), which handles the low-level plumbing (sending wheel-velocity JSON commands to the firmware, unit conversions, safety clamps). The subclass only needs to say *how* the rover should move:

```python
class YoloFollower(Rover):
    TARGET_OBJECT = "bottle"  # the COCO object name we drive toward
```

The constructor sets up the pieces this behavior needs: how close counts as "arrived," where the camera is, and the YOLO pose estimator that will turn camera frames into 3D positions:

```python
    def __init__(self, stop_tolerance_m=0.25):
        super().__init__()

        # How far away from the object (in meters) the rover should stop
        self.stop_tolerance_m = stop_tolerance_m

        # Camera Stream Address
        self.img_addr = "http://192.168.50.123:80/capture"

        # The locally downloaded YOLOv8n weights that ship with the YOLO_agent folder
        model_path = Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"

        # Create our YOLO 3D pose estimator (imgsz=960 finds smaller/farther objects)
        self.estimator = YOLOPoseEstimator(
            model_path=model_path,
            imgsz=960,
            verbose=True,  # set False if you only want data
        )
```

Steering a rover toward a point isn't a single number — it needs two independent controllers: one to manage forward speed based on distance, and one to manage turning based on how far off to the side the object is. Both are PID controllers (`rover_control/pid.py`), which continuously correct their output based on the current error rather than just reacting once:

```python
        # PID on the forward distance to the object (+Z is forward), output is forward speed in m/s
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)

        # PID on the sideways offset of the object (+X is right), output is a turning speed in m/s
        self.heading_pid = PID(kp=0.6, ki=0.0, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # Remember when we last ran the controller so the PIDs get a real time step
        self._last_pid_time = time.perf_counter()
```

`wheel_speeds_for()` is where a 3D position turns into actual motor commands. First it works out how much time has passed since the last control step, which PID math needs to compute proper derivatives and integrals:

```python
    def wheel_speeds_for(self, position):
        # Measure the time since the last control step for the PID math
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # The error is how much farther than the stop tolerance the object is (+Z: forward)
        distance_error = position[2] - self.stop_tolerance_m

        # Once we're inside the tolerance, stop and clear the controllers
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]
```

If the rover isn't close enough yet, each PID controller converts its piece of the error (how far forward, how far sideways) into a speed, and the two are combined the way a differential-drive robot always steers: add turn to one wheel, subtract it from the other, so the rover curves toward the object as it drives forward:

```python
        # Forward speed comes from the distance PID, turning comes from the sideways offset PID
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(position[0], dt)

        # Mix forward and turn for a differential drive (object to the right -> left wheel speeds
        # up)
        speed = [forward + turn, forward - turn]
```

Those speeds are in linear meters/second, but the rover's firmware and the `Rover` base class's clamping logic work in angular wheel speed, so they get converted and clamped to the rover's physical limits before being returned:

```python
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

`update()` is the main loop that ties it all together. Each pass, it pulls one frame from the ESP32 camera and runs it through the pose estimator to get every detected object's 3D position, same as the plain 3D-pose example:

```python
        while True:
            # Grab one JPEG still from the ESP32 camera, then find the object and its 3D pose
            try:
                frame = self.estimator.get_frame_from_http(self.img_addr)
            except Exception as error:  # noqa: BLE001 -- one bad frame should not stop the rover mid-drive
                print(f"Camera error: {error}")
                continue
            annotated, detections = self.estimator.process(frame)
```

Out of everything the network detects in a frame, only the target object matters, and if several show up (say, more than one bottle) the rover aims at whichever is nearest, since that's the one most likely to be the intended target:

```python
            # Keep only our target object, and aim at the nearest one if several show up
            targets = [d for d in detections if d["name"] == self.TARGET_OBJECT]
            if targets:
                nearest = min(targets, key=lambda d: d["position"][2])
                print("Pose : ", nearest["position"])
                wheel_speeds = self.wheel_speeds_for(nearest["position"])
            else:
                # Object not in sight, so hold still
                wheel_speeds = [0.0, 0.0]
```

Finally, the loop sends the computed wheel speeds down to the firmware and sleeps just long enough to keep commands flowing at the rate the firmware expects, using the same "sleep the leftover time" pattern seen in the vision-only examples:

```python
            self.write_real_velocities(wheel_speeds)
            # Sleep only the leftover time so commands go out at the firmware's rate
            self.sleep_to_command_rate()

        # Make sure the rover doesn't keep rolling after we quit
        self.stop()
        cv2.destroyAllWindows()
```

## See also

- [YOLO_agent/](../../YOLO_agent/) — the detector and pose estimator modules this example depends on
- [rover_control/](../../rover_control/) — the Rover base class and other examples
- [Back to README](../../README.md)
