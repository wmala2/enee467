import time
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOExtractor:
    def __init__(
        self,
        model_path,
        confidence=0.5,
        imgsz=640,
        verbose=False,
    ):
        ### Load the YOLO network weights from a local .pt file ###
        self.model = YOLO(model_path)

        ### Detection configuration ###
        # imgsz is the size the image is scaled to before the network sees it:
        # bigger finds smaller/farther objects but runs slower (640 ~17 Hz, 960 ~9 Hz on CPU)
        self.confidence = confidence
        self.imgsz = imgsz
        self.verbose = verbose

        ### FPS tracking between process() calls ###
        self._fps = 0.0
        self._last_process_time = None

    def get_frame_from_http(self, http_addr):
        """
        Fetch and decode a JPEG still image from an HTTP endpoint,
        like the ESP32 camera's /capture page.
        """
        with urllib.request.urlopen(http_addr, timeout=2.0) as response:
            data = np.frombuffer(response.read(), np.uint8)

        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def process(self, frame):
        """
        Single-step detection call:
        - measures the loop frame rate
        - runs the YOLO network on one frame
        - converts the raw output into simple dictionaries
        - optionally renders an annotated frame
        - returns (frame, detections)
        """
        ### Measure the real frame rate from one process() call to the next ###
        now = time.perf_counter()
        if self._last_process_time is not None:
            self._fps = 1.0 / (now - self._last_process_time)
        self._last_process_time = now

        if frame is None:
            return None, []

        ### Run the neural network on this frame ###
        results = self.model(frame, conf=self.confidence, imgsz=self.imgsz, verbose=False)[0]

        ### Convert the raw network output into a simple list of dictionaries ###
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append({
                "name": results.names[int(box.cls)],  # what the object is
                "confidence": float(box.conf),  # how sure the network is (0-1)
                "box": [x1, y1, x2, y2],  # corners of the bounding box (pixels)
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],  # middle of the box (pixels)
            })

        ### Optional visualization with boxes, labels, and the frame rate ###
        annotated = None
        if self.verbose:
            annotated = results.plot()
            cv2.putText(
                annotated,
                f"FPS: {self._fps:.1f}",
                (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        return annotated, detections

    def get_fps(self):
        """
        Standalone getter: returns the measured frame rate (Hz) of the process() loop.
        """
        return self._fps

    def sleep_to_fps(self, target_fps):
        """
        Sleep only the time left over in this frame so the loop runs at target_fps,
        no matter how long the detection itself took.
        """
        if self._last_process_time is None:
            return

        elapsed = time.perf_counter() - self._last_process_time
        remaining = (1.0 / target_fps) - elapsed
        if remaining > 0:
            time.sleep(remaining)
