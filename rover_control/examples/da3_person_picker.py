"""Click on a person in a DA3 depth-colored view, then follows just that one.

Like da3_object_movement.py, this asks a GPU server (depth_anything_server) for a full depth
map of the scene instead of using a known marker size. But it doesn't just drive at the first
person YOLO happens to see. Instead it shows every person currently visible, overlaid on the
depth map in color, and waits for a click before doing anything -- then keeps following that
specific person by their tracked ID, not just "whoever's closest right now."
"""

# External Libraries
import argparse
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

PERSON_CLASS_ID = 0  # COCO class index for "person"


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


def depth_to_colormap(depth_map, max_display_depth=10.0):
    """Turns a metric depth map (meters per pixel) into a viewable color image, warm = close."""
    depth_clipped = np.clip(depth_map, 0, max_display_depth)
    depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)


class PersonPicker(Rover):
    """A rover that shows every visible person on a depth map, waits for a click, then follows
    whoever was chosen using GPU depth + PID control.
    """

    ROVER_CAMERA_IP = "192.168.50.123:80"
    SERVER_IP = "192.168.50.155:5000"
    CONFIDENCE_THRESH = 0.80
    FOCAL_LENGTH_PX = 371.818  # measured once for this camera; used to turn pixels into meters
    PICKER_WINDOW = "Choose a person to follow (click their box)"
    FOLLOW_WINDOW = "Following"

    def __init__(self, server_ip=SERVER_IP, rover_camera_ip=ROVER_CAMERA_IP, stop_tolerance_m=0.60):
        super().__init__()

        # How close (in meters) the rover should get to the chosen person before it stops.
        self.stop_tolerance_m = stop_tolerance_m

        # YOLOExtractor finds and tracks people in each frame; RoverNavigationClient asks the
        # GPU server for a depth map; DepthWorker runs that pair continuously in the background.
        yolo_model_path = (
            Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"
        )
        self.extractor = YOLOExtractor(model_path=yolo_model_path, imgsz=640, verbose=False)
        # verbose=True here would be handy for error text, but it also makes get_metric_depth()
        # pop up its own debug window (RoverNavigationClient._show_debug_window) - since this
        # class calls it from DepthWorker's background thread while this script's own windows
        # are drawn on the main thread, two threads fight over OpenCV's GUI state and this
        # script's own picker/following windows stop rendering. Keep this False.
        self.client = RoverNavigationClient(server_url="http://" + server_ip, verbose=False)
        self._depth_worker = DepthWorker(self.client, rover_camera_ip)

        # A PID controller is a small feedback loop: measure how wrong you are (the error), and
        # nudge your output based on how big that error is and how fast it's changing. We need
        # one PID to control forward speed and a second one to control steering.
        self.distance_pid = PID(kp=0.75, ki=0.10, kd=0.15, output_limit=self.MAX_VELOCITY)
        self.heading_pid = PID(kp=0.60, ki=0.00, kd=0.05, output_limit=self.MAX_VELOCITY / 2.0)
        self._last_pid_time = time.perf_counter()

        # None until the student clicks someone; the picker window stays up while this is None.
        self.chosen_id = None
        # This frame's boxes, keyed by track ID -- _on_click reads this to figure out which
        # person (if any) a click landed on.
        self._visible_boxes = {}

    def _on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for track_id, (x1, y1, x2, y2) in self._visible_boxes.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.chosen_id = track_id
                print(f"Following person #{track_id}")
                return

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

    def wheel_speeds_for(self, pseudo_pose):
        """Turns one (X, _, Z) position estimate into a pair of [left, right] wheel speeds."""
        now = time.perf_counter()
        dt = now - self._last_pid_time
        self._last_pid_time = now

        distance_error = pseudo_pose[2] - self.stop_tolerance_m
        if distance_error <= 0:
            self.distance_pid.reset()
            self.heading_pid.reset()
            return [0.0, 0.0]

        forward = self.distance_pid.update(distance_error, dt)
        turn = self.heading_pid.update(pseudo_pose[0], dt)
        speed = [forward + turn, forward - turn]

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

    def _show_picker(self, depth_map, people):
        # Draw every visible person's box + ID on the depth colormap, and remember where each
        # box is so _on_click can tell which one got clicked.
        colormap = depth_to_colormap(depth_map)
        self._visible_boxes = {}
        for person in people:
            x1, y1, x2, y2 = map(int, person["box"])
            self._visible_boxes[person["id"]] = (x1, y1, x2, y2)
            cv2.rectangle(colormap, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                colormap,
                f"#{person['id']}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        cv2.imshow(self.PICKER_WINDOW, colormap)

    def _show_following(self, frame, person, distance_z):
        annotated = frame.copy()
        x1, y1, x2, y2 = map(int, person["box"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            f"#{person['id']}: {distance_z:.2f}m",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow(self.FOLLOW_WINDOW, annotated)

    def update(self):
        """The main loop: track every person, show what there is to choose from or what's being
        followed, decide how fast to drive, send the command, repeat.
        """
        print("Click a person's box in the depth view to choose who the rover should follow.")
        print("Press 'q' inside a video window to abort.")
        cv2.namedWindow(self.PICKER_WINDOW)
        cv2.setMouseCallback(self.PICKER_WINDOW, self._on_click)
        picker_open = True

        try:
            while True:
                # Grab whatever the background DepthWorker has most recently fetched, instead of
                # waiting on the GPU server ourselves.
                frame, depth_map = self._depth_worker.latest()
                if frame is None or depth_map is None:
                    time.sleep(0.05)
                    continue

                # Track every visible person (not just one) so their IDs stay stable whether
                # we're still picking or already following.
                people = self.extractor.track(frame, classes=[PERSON_CLASS_ID])
                people = [p for p in people if p["confidence"] > self.CONFIDENCE_THRESH]

                wheel_speeds = [0.0, 0.0]  # default: sit still unless we're actively following

                if self.chosen_id is None:
                    self._show_picker(depth_map, people)
                else:
                    if picker_open:  # only needs closing once, right when a choice is made
                        cv2.destroyWindow(self.PICKER_WINDOW)
                        picker_open = False

                    match = next((p for p in people if p["id"] == self.chosen_id), None)
                    if match is None:
                        # Lost them (left frame, occluded, etc.) -- stop and keep watching for
                        # the tracker to hand back the same ID. It won't always: a long
                        # occlusion or someone leaving and re-entering frame can get a new ID,
                        # in which case this rover just waits here until you restart the script.
                        print(f"Lost person #{self.chosen_id}, waiting for them to reappear...")
                    else:
                        distance_z = self.calculate_filtered_distance(match["box"], depth_map)
                        frame_width = frame.shape[1]
                        box_center_x = (match["box"][0] + match["box"][2]) / 2.0
                        pixel_offset_x = box_center_x - (frame_width / 2.0)
                        distance_x = (pixel_offset_x * distance_z) / self.FOCAL_LENGTH_PX

                        wheel_speeds = self.wheel_speeds_for([distance_x, 0.0, distance_z])
                        self._show_following(frame, match, distance_z)

                self.write_real_velocities(wheel_speeds)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                self.sleep_to_command_rate()

        except KeyboardInterrupt:
            print("\nManual override. Stopping.")
        finally:
            # Always stop the motors, the background thread, and the windows -- never leave a
            # rover rolling.
            self._depth_worker.stop()
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default=PersonPicker.SERVER_IP,
        help="GPU depth server's host:port, no http:// (default: %(default)s)",
    )
    parser.add_argument(
        "--camera-ip",
        default=PersonPicker.ROVER_CAMERA_IP,
        help="Rover's ESP32 camera host:port, no http:// or /capture (default: %(default)s)",
    )
    parser.add_argument(
        "--stop-distance-m",
        type=float,
        default=0.60,
        help="How close to get to the chosen person before stopping (default: %(default)s)",
    )
    args = parser.parse_args()

    picker = PersonPicker(
        server_ip=args.server,
        rover_camera_ip=args.camera_ip,
        stop_tolerance_m=args.stop_distance_m,
    )
    picker.update()
