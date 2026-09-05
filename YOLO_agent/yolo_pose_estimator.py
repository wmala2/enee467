import os

# If the depth model is already in the local Hugging Face cache, force offline mode so it loads
# straight from disk with no network calls. This lets the YOLO 3D-pose examples run on a network
# with no internet (like the rover's BaleNet). The first run still needs internet to download it once.
_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
_hf_hub = os.environ.get("HF_HUB_CACHE", os.path.join(_hf_home, "hub"))
if os.path.isdir(os.path.join(_hf_hub, "models--" + _DEPTH_MODEL.replace("/", "--"))):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline

from YOLO_agent.depth_client import DepthServerClient
from YOLO_agent.YOLO_extractor import YOLOExtractor

# Default camera calibration (same ESP32 numbers the ArUco estimator uses)
camera_matrix_default = np.array(
    [[372.69444659, 0.0, 321.25586034], [0.0, 370.94214518, 238.16525187], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# A small, metric (meters) monocular depth model that runs on a laptop CPU
depth_model_default = _DEPTH_MODEL


class YOLOPoseEstimator:
    """
    Gives every YOLO detection a 3D position (X, Y, Z) in meters, with no ArUco tag.
    It pairs YOLO (what + where in the image) with a depth model (how far each pixel is),
    then back-projects the box center into the same (X right, Y down, Z forward) frame
    the ArUco tracker reports.
    """

    def __init__(
        self,
        model_path,
        camera_matrix=camera_matrix_default,
        depth_model=depth_model_default,
        confidence=0.5,
        imgsz=640,
        verbose=False,
        depth_server_url=None,
    ):
        ### YOLO does the object detection (reuse the class the other examples use) ###
        self.detector = YOLOExtractor(
            model_path=model_path,
            confidence=confidence,
            imgsz=imgsz,
            verbose=verbose,
        )

        ### Pull the focal lengths and image center out of the calibration matrix ###
        self.fx = float(camera_matrix[0, 0])
        self.fy = float(camera_matrix[1, 1])
        self.cx = float(camera_matrix[0, 2])
        self.cy = float(camera_matrix[1, 2])

        ### Depth backend: remote GPU server if a URL is given, otherwise local CPU/GPU ###
        if depth_server_url is not None:
            self._remote_depth = DepthServerClient(depth_server_url)
            self.depth_pipe = None
            print(f"Depth backend: remote server at {depth_server_url}")
        else:
            self._remote_depth = None
            device = 0 if torch.cuda.is_available() else -1
            self.depth_pipe = pipeline("depth-estimation", model=depth_model, device=device)
            print(f"Depth backend: local model ({depth_model})")

        self.verbose = verbose

    def get_frame_from_http(self, http_addr):
        """
        Fetch one JPEG still from the ESP32 camera (delegates to the YOLO extractor).
        """
        return self.detector.get_frame_from_http(http_addr)

    def _depth_map_meters(self, frame):
        """
        Return a meters-per-pixel depth map the same size as frame.
        Uses the remote GPU server if one was configured, otherwise runs locally.
        """
        if self._remote_depth is not None:
            return self._remote_depth.get_depth(frame)

        # Local path: depth model expects RGB PIL image, OpenCV gives us BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.depth_pipe(Image.fromarray(rgb))

        # predicted_depth is the raw metric tensor; squeeze it down to a 2D numpy array
        depth = result["predicted_depth"].squeeze().cpu().numpy()

        # The model works at its own resolution, so stretch the map back to the frame size
        height, width = frame.shape[:2]
        return cv2.resize(depth, (width, height))

    def process(self, frame):
        """
        Single-step call: detect objects, look up each one's depth, and back-project to (X, Y, Z).
        Returns (annotated_frame, detections) where each detection gains "position" and "distance".
        """
        if frame is None:
            return None, []

        ### Detect objects and (when verbose) get the annotated image back ###
        annotated, detections = self.detector.process(frame)

        ### One depth pass for the whole frame, shared by every detection ###
        depth_map = self._depth_map_meters(frame)
        height, width = depth_map.shape[:2]

        ### Turn each box center into a real-world position ###
        for detection in detections:
            # Pixel coordinates of the box center, clamped so we never index off the image
            u = int(min(max(detection["center"][0], 0), width - 1))
            v = int(min(max(detection["center"][1], 0), height - 1))

            # Depth straight out of the camera at that pixel (meters, +Z forward)
            Z = float(depth_map[v, u])

            # Back-project the pixel to meters using the pinhole camera model
            X = (u - self.cx) * Z / self.fx  # +X to the right
            Y = (v - self.cy) * Z / self.fy  # +Y downward

            detection["position"] = [X, Y, Z]  # X, Y, Z in meters
            detection["distance"] = float(np.sqrt(X * X + Y * Y + Z * Z))

        return annotated, detections

    def get_object_position(self, frame, name):
        """
        Standalone getter: return the (X, Y, Z) of the closest object called `name`, or None.
        """
        _, detections = self.process(frame)

        # Keep only the detections whose label matches what we asked for
        matches = [d for d in detections if d["name"] == name]
        if not matches:
            return None

        # When several match, pick the nearest one (smallest Z)
        nearest = min(matches, key=lambda d: d["position"][2])
        return nearest["position"]
