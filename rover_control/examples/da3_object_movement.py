"""
Multi-Rover Object-Following Autonomous Control Loop

Combines the peak-density GPU depth telemetry pipeline with differential-drive
PID velocity wheels tracking to follow an object (e.g., a person).
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
    """
    Background thread that continuously fetches frames and depth maps so the
    control loop never blocks waiting for GPU inference (~250 ms per call).
    The main loop calls latest() to get the most recent frame+depth pair instantly.
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
        while self._running:
            frame = self._client.fetch_rover_frame(self._rover_ip)
            if frame is None:
                time.sleep(0.05)
                continue
            depth = self._client.get_metric_depth(frame)
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
    TARGET_CLASS = "person"  # Target item label to track and follow
    ROVER_CAMERA_IP = "192.168.50.123:80"
    SERVER_IP = "192.168.50.155:5000"
    CONFIDENCE_THRESH = 0.80
    FOCAL_LENGTH_PX = 371.818  # Calibrated pixel focal length metric

    def __init__(self, stop_tolerance_m=0.60):
        """
        Initializes the rover control loop.
        stop_tolerance_m: The distance (meters) to maintain from the object.
        """
        super().__init__()
        self.stop_tolerance_m = stop_tolerance_m

        # Instantiate your custom peak-isolated vision clients
        yolo_model_path = (
            Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"
        )
        self.extractor = YOLOExtractor(model_path=yolo_model_path, imgsz=640, verbose=False)
        self.client = RoverNavigationClient(server_url="http://" + self.SERVER_IP, verbose=False)
        self._depth_worker = DepthWorker(self.client, self.ROVER_CAMERA_IP)

        # PID controllers matching your historical ArUco hardware configurations
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)
        self.heading_pid = PID(kp=0.60, ki=0.00, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)

        self._last_pid_time = time.perf_counter()

    def calculate_filtered_distance(self, corners, metric_depth_map):
        """Standard Peak-Density Isolation to return precise target center range."""
        x1, y1, x2, y2 = map(int, corners)
        box_depths = metric_depth_map[y1:y2, x1:x2].flatten()
        if len(box_depths) == 0:
            return 0.0
        box_depths = box_depths[box_depths > 0.1]
        if len(box_depths) == 0:
            return 0.0

        bucket_width = 0.05
        bins = np.arange(0.0, 10.0 + bucket_width, bucket_width)
        counts, bin_edges = np.histogram(box_depths, bins=bins)

        max_bucket_idx = np.argmax(counts)
        peak_center = (bin_edges[max_bucket_idx] + bin_edges[max_bucket_idx + 1]) / 2.0

        tolerance = 0.15
        filtered_pixels = box_depths[
            (box_depths >= peak_center - tolerance) & (box_depths <= peak_center + tolerance)
        ]

        return float(peak_center) if len(filtered_pixels) == 0 else float(np.mean(filtered_pixels))

    def compute_wheel_speeds(self, pseudo_pose):
        """Mixes forward velocity and steering trim via standard differential control."""
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        # error = Forward range - stop point threshold
        distance_error = pseudo_pose[2] - self.stop_tolerance_m

        # If inside stop tolerance boundary, park completely
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]

        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(pseudo_pose[0], dt)

        # Differential wheel drive blending calculation
        speed = [forward + turn, forward - turn]

        # Standard hardware pseudo-twist mapping configurations
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
        # Explicit debugging context active as requested
        print(f"\n[VERBOSE SYSTEM ACTIVE] Seeking target label: '{self.TARGET_CLASS}'")
        print(f"Maintaining distance safety margin: {self.stop_tolerance_m} meters.")
        print("Press 'q' inside the video viewport window to abort.")

        try:
            while True:
                # Non-blocking — depth worker keeps this fresh in the background
                frame, depth_map = self._depth_worker.latest()
                if frame is None:
                    time.sleep(0.05)
                    continue

                _, w, _ = frame.shape
                annotated, detections = self.extractor.process(frame)

                # Filter out the specific target type
                valid_targets = [
                    d
                    for d in detections
                    if d["name"] == self.TARGET_CLASS and d["confidence"] > self.CONFIDENCE_THRESH
                ]

                wheel_speeds = [0.0, 0.0]  # Default to zero velocity state if empty

                if len(valid_targets) > 0:
                    if depth_map is not None:
                        # Process first matching prioritized target
                        target = valid_targets[0]
                        corners = target["box"]
                        x1, y1, x2, y2 = map(int, corners)

                        # 1. Forward distance from peak histogram cluster
                        distance_z = self.calculate_filtered_distance(corners, depth_map)

                        # 2. Sideways tracking error synthesis
                        box_center_x = (x1 + x2) / 2.0
                        image_center_x = w / 2.0
                        pixel_offset_x = box_center_x - image_center_x

                        # Pinpoint horizontal metric offset location
                        distance_x = (pixel_offset_x * distance_z) / self.FOCAL_LENGTH_PX

                        # Synthesize structural pseudo-pose layout array matching tracking history
                        pseudo_pose = [distance_x, 0.0, distance_z]

                        print(
                            f"[TRACKING VERBOSE] Box Center X: {box_center_x:.1f}px | "
                            f"X Offset: {distance_x:.2f}m | Z Depth: {distance_z:.2f}m"
                        )

                        # Compile steering rates
                        wheel_speeds = self.compute_wheel_speeds(pseudo_pose)
                        print(
                            f"                  -> Computed Commands: Left={wheel_speeds[0]:.2f}, "
                            f"Right={wheel_speeds[1]:.2f}"
                        )

                        # Visual feedback layer additions
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
                    print("[TRACKING VERBOSE] Scanning scene... No valid targets identified.")
                    self.distance_pid.reset()
                    self.heading_pid.reset()

                # Ensure 'annotated' is a valid, non-empty 3D image matrix. Fallback to raw frame if
                # it's broken.
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

                # Dispatch control vectors straight to firmware
                self.write_real_velocities(wheel_speeds)
                self.sleep_to_command_rate()

        except KeyboardInterrupt:
            print("\nManual override interception. Commencing soft stop sequence.")
        finally:
            self._depth_worker.stop()
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize deployment base with 0.60m stopping tolerance buffer zone
    follower = ObjectFollower(stop_tolerance_m=0.60)
    follower.update()
