"""Follows one named object (default: a person) using a GPU-computed depth map instead of a tag.

Unlike the ArUco/YOLO followers, this one doesn't get distance for free from a known marker size
or a specialized 3D model -- it asks a GPU server (depth_anything_server) for a full depth map of
the scene, then reads the distance to whatever YOLO box it's tracking straight out of that map.
"""

# External Libraries
from pathlib import Path
import threading
import time

import cv2
import numpy as np

# Local Network Client & Vision Pipelines
from depth_anything_server.depth_client import RoverNavigationClient

# Local Core Rover Framework
from rover_control import conversions
from rover_control.pid import PID
from rover_control.rover import Rover
from YOLO_agent.YOLO_extractor import YOLOExtractor


class DepthWorker:
    """Fetches camera frames and depth maps on a background thread.

    Asking the GPU server for a depth map takes about 250 ms -- far too slow to do inline in a
    10 Hz control loop. Instead, this class fetches continuously in the background, and the main
    loop just grabs whatever the newest frame+depth pair happens to be with `latest()`, never
    waiting on the network.
    """

    def __init__(self, client: RoverNavigationClient, rover_ip: str):
        self._client = client
        self._rover_ip = rover_ip
        self._lock = threading.Lock()
        self._frame = None
        self._depth = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # This loop runs forever on its own thread, constantly refreshing self._frame/_depth.
        while self._running:
            frame = self._client.fetch_rover_frame(self._rover_ip)
            if frame is None:
                time.sleep(0.05)
                continue
            depth = self._client.get_metric_depth(frame)
            # Hold the lock only long enough to swap in the new pair, so latest() never blocks
            # for the ~250 ms it took to compute them.
            with self._lock:
                self._frame = frame
                self._depth = depth

    def latest(self):
        """Returns (frame, depth_map) — both from the same capture, never blocking."""
        with self._lock:
            return self._frame, self._depth

    def stop(self):
        self._running = False


class ObjectFollower(Rover):
    """A rover that finds one named object with YOLO and drives up to it using GPU depth."""

    TARGET_CLASS = "person"  # any COCO class name works here
    ROVER_CAMERA_IP = "192.168.50.123:80"
    SERVER_IP = "192.168.50.155:5000"
    CONFIDENCE_THRESH = 0.80
    FOCAL_LENGTH_PX = 371.818  # measured once for this camera; used to turn pixels into meters

    def __init__(self, stop_tolerance_m=0.60):
        super().__init__()

        # How close (in meters) the rover should get to the object before it stops driving.
        self.stop_tolerance_m = stop_tolerance_m

        # YOLOExtractor finds the object in each frame; RoverNavigationClient asks the GPU
        # server for a depth map; DepthWorker runs that pair continuously in the background.
        yolo_model_path = (
            Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"
        )
        self.extractor = YOLOExtractor(model_path=yolo_model_path, imgsz=640, verbose=False)
        self.client = RoverNavigationClient(server_url="http://" + self.SERVER_IP, verbose=False)
        self._depth_worker = DepthWorker(self.client, self.ROVER_CAMERA_IP)

        # A PID controller is a small feedback loop: measure how wrong you are (the error), and
        # nudge your output based on how big that error is and how fast it's changing. We need
        # one PID to control forward speed and a second one to control steering.
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)
        self.heading_pid = PID(kp=0.60, ki=0.00, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        # PID math needs to know how much time passed since the last update, so remember it here.
        self._last_pid_time = time.perf_counter()

    def calculate_filtered_distance(self, corners, metric_depth_map):
        """Estimates one clean distance (in meters) to whatever's inside a YOLO box.

        A depth map's raw pixels inside a bounding box are noisy: some belong to the target,
        some belong to background poking through gaps around it. This finds the most common
        depth value in the box (its "peak") and averages only the pixels close to that peak,
        which throws out the background outliers.
        """
        x1, y1, x2, y2 = map(int, corners)
        box_depths = metric_depth_map[y1:y2, x1:x2].flatten()
        if len(box_depths) == 0:
            return 0.0
        # Depths near zero are usually lens glare or sensor noise, not a real surface.
        box_depths = box_depths[box_depths > 0.1]
        if len(box_depths) == 0:
            return 0.0

        # Bucket every depth reading into 5 cm bins, then find the bin with the most pixels in
        # it -- that's almost always the target object rather than the background around it.
        bucket_width = 0.05
        bins = np.arange(0.0, 10.0 + bucket_width, bucket_width)
        counts, bin_edges = np.histogram(box_depths, bins=bins)

        max_bucket_idx = np.argmax(counts)
        peak_center = (bin_edges[max_bucket_idx] + bin_edges[max_bucket_idx + 1]) / 2.0

        # Average only the pixels within 15 cm of that peak, for a cleaner final distance.
        tolerance = 0.15
        filtered_pixels = box_depths[
            (box_depths >= peak_center - tolerance) & (box_depths <= peak_center + tolerance)
        ]

        return float(peak_center) if len(filtered_pixels) == 0 else float(np.mean(filtered_pixels))

    def wheel_speeds_for(self, pseudo_pose):
        """Turns one (X, _, Z) position estimate into a pair of [left, right] wheel speeds."""
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # The "error" here is simple: how much farther away is the object than where we want to
        # stop?
        distance_error = pseudo_pose[2] - self.stop_tolerance_m

        # Already close enough? Stop, and reset the PIDs so old error doesn't carry into next
        # time.
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]

        # Feed each error into its own PID to get a forward speed and a turning speed.
        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(pseudo_pose[0], dt)

        # A differential-drive rover turns by spinning its two wheels at different speeds: add
        # the turn amount to one side and subtract it from the other.
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

        return self.clamp(speed, min_pos, max_pos)

    def update(self):
        """The main loop: look, decide how fast to drive, send the command, repeat."""
        print(f"\n[VERBOSE SYSTEM ACTIVE] Seeking target label: '{self.TARGET_CLASS}'")
        print(f"Maintaining distance safety margin: {self.stop_tolerance_m} meters.")
        print("Press 'q' inside the video viewport window to abort.")

        try:
            while True:
                # Grab whatever the background DepthWorker has most recently fetched, instead of
                # waiting on the GPU server ourselves.
                frame, depth_map = self._depth_worker.latest()
                if frame is None:
                    time.sleep(0.05)
                    continue

                _, w, _ = frame.shape
                annotated, detections = self.extractor.process(frame)

                # Keep only confident detections of our target class.
                valid_targets = [
                    d
                    for d in detections
                    if d["name"] == self.TARGET_CLASS and d["confidence"] > self.CONFIDENCE_THRESH
                ]

                wheel_speeds = [0.0, 0.0]  # default: sit still unless we find a target below

                if len(valid_targets) > 0:
                    if depth_map is not None:
                        # If several match, just follow the first one YOLO reported.
                        target = valid_targets[0]
                        corners = target["box"]
                        x1, y1, x2, y2 = map(int, corners)

                        # Forward distance comes straight from the depth map.
                        distance_z = self.calculate_filtered_distance(corners, depth_map)

                        # Sideways offset has to be worked out by hand: how far the box's
                        # center sits from the image's center, in pixels, converted to meters
                        # using the camera's focal length and how far away the object already is.
                        box_center_x = (x1 + x2) / 2.0
                        image_center_x = w / 2.0
                        pixel_offset_x = box_center_x - image_center_x
                        distance_x = (pixel_offset_x * distance_z) / self.FOCAL_LENGTH_PX

                        # Package the two numbers the same way the ArUco/YOLO followers do, so
                        # the same PID-based steering code below can use either.
                        pseudo_pose = [distance_x, 0.0, distance_z]

                        print(
                            f"[TRACKING VERBOSE] Box Center X: {box_center_x:.1f}px | "
                            f"X Offset: {distance_x:.2f}m | Z Depth: {distance_z:.2f}m"
                        )

                        wheel_speeds = self.wheel_speeds_for(pseudo_pose)
                        print(
                            f"                  -> Computed Commands: Left={wheel_speeds[0]:.2f}, "
                            f"Right={wheel_speeds[1]:.2f}"
                        )

                        # Draw the box and current distance on the preview window.
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"FOLLOWING {self.TARGET_CLASS.upper()}: {distance_z:.2f}m"
                        cv2.putText(
                            annotated,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                else:
                    # Nothing to follow this frame, so hold still instead of driving blind.
                    print("[TRACKING VERBOSE] Scanning scene... No valid targets identified.")
                    self.distance_pid.reset()
                    self.heading_pid.reset()

                # `annotated` can come back empty/malformed on a bad detector frame; fall back to
                # the plain camera frame rather than crashing the preview window.
                if (
                    annotated is not None
                    and isinstance(annotated, np.ndarray)
                    and annotated.ndim == 3
                    and annotated.size > 0
                ):
                    cv2.imshow("Object Follower Active Stream", annotated)
                else:
                    cv2.imshow("Object Follower Active Stream", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Actually send the command, then wait just long enough to hit our fixed control
                # rate.
                self.write_real_velocities(wheel_speeds)
                self.sleep_to_command_rate()

        except KeyboardInterrupt:
            print("\nManual override interception. Commencing soft stop sequence.")
        finally:
            # Always stop the motors, the background thread, and the windows -- never leave a
            # rover rolling.
            self._depth_worker.stop()
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    follower = ObjectFollower(stop_tolerance_m=0.60)
    follower.update()
