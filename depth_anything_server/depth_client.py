# Internal Imports
import cv2

# External Imports
import numpy as np
import requests


class RoverNavigationClient:
    def __init__(self, server_url: str, timeout: float = 5.0, verbose: bool = False):
        self.depth_url = f"{server_url.rstrip('/')}/depth"
        # Reuse one TCP connection across all requests — avoids per-frame handshake overhead
        self.session = requests.Session()
        self.timeout = timeout
        self.verbose = verbose  # Toggle to automatically render the depth window

    def fetch_rover_frame(self, rover_ip: str) -> np.ndarray:
        """Captures a snapshot from the rover to minimize thermal load."""
        capture_url = f"http://{rover_ip}/capture"
        try:
            response = self.session.get(capture_url, timeout=3.0)
            response.raise_for_status()

            img_arr = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            if self.verbose:
                print(f"Error communicating with rover {rover_ip}: {e}")
            return None

    def get_metric_depth(self, frame: np.ndarray) -> np.ndarray:
        """Sends the frame to the GPU server and receives raw float32 meters."""
        # Quality 90 keeps edges sharp enough for accurate depth without bloating the payload
        ok, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG compression failed.")

        try:
            resp = self.session.post(
                self.depth_url,
                data=jpeg_buf.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=self.timeout,
            )
            resp.raise_for_status()

            # Server sends raw float32 bytes; width/height come back in custom headers
            h = int(resp.headers["X-Depth-Height"])
            w = int(resp.headers["X-Depth-Width"])
            depth_meters = np.frombuffer(resp.content, dtype=np.float32).reshape(h, w)

            # Model may output at a different resolution than the input — resize to match
            frame_h, frame_w = frame.shape[:2]
            if (h, w) != (frame_h, frame_w):
                # INTER_NEAREST preserves hard depth edges (no blending across object boundaries)
                depth_meters = cv2.resize(
                    depth_meters, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
                )

            # If verbose is active, show the live visual feedback window
            if self.verbose:
                self._show_debug_window(frame, depth_meters)

            return depth_meters
        except Exception as e:
            if self.verbose:
                print(f"Server inference failed: {e}")
            return None

    def _show_debug_window(self, frame: np.ndarray, depth_map: np.ndarray):
        """Internal helper to render a side-by-side feed."""
        # Normalize metric depth (0-10 meters capped) to standard 0-255 grayscale
        max_dist = 10.0
        depth_clipped = np.clip(depth_map, 0, max_dist)
        depth_visual = ((1.0 - (depth_clipped / max_dist)) * 255).astype(np.uint8)

        # Colorize it so it looks spectacular for the students
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

        # Stack original camera view and depth view side-by-side
        combined_view = cv2.hconcat([frame, depth_colormap])

        cv2.imshow("Rover Telemetry (Left: RGB | Right: Depth)", combined_view)
        cv2.waitKey(1)  # Keeps the window responsive

    def get_object_distance(self, yolo_box, depth_map) -> float:
        """Filters out background outliers and returns target distance."""
        x1, y1, x2, y2 = map(int, yolo_box)
        # Crop the depth map to just the pixels inside the YOLO bounding box
        box_depths = depth_map[y1:y2, x1:x2].flatten()

        if len(box_depths) == 0:
            return 0.0
        # Drop near-zero readings caused by lens glare or sensor noise
        box_depths = box_depths[box_depths > 0.1]

        # IQR filter: keep only the middle 50% of depth values to remove background outliers
        q25, q75 = np.percentile(box_depths, [25, 75])
        iqr = q75 - q25
        filtered_pixels = box_depths[
            (box_depths >= q25 - 1.5 * iqr) & (box_depths <= q75 + 1.5 * iqr)
        ]

        if len(filtered_pixels) == 0:
            return float(np.median(box_depths))
        return float(np.mean(filtered_pixels))
