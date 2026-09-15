"""Asks a local vision LLM what object is in one camera frame.

This is a lighter-weight alternative to a trained detector like YOLO: instead of a model built
to recognize a fixed list of classes, a general vision-language model looks at the picture and
answers in plain words, which we force into a small structured JSON reply.
"""

# External Libraries
import json
import os
import time
from typing import ClassVar

import cv2
import numpy as np
import ollama


class ObjectIdentifier:
    """Asks a vision model to name the main object in one image, YOLO-detection style."""

    # The exact JSON shape we force the model to reply with
    RESPONSE_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {"object": {"type": "string"}},
        "required": ["object"],
    }

    def __init__(self, model="moondream", keep_alive="10m"):
        # moondream is a tiny (1.8B) vision model - the only one fast enough on a laptop CPU
        self.model = model
        self.keep_alive = keep_alive

        # Using most of the CPU cores roughly halves the response time
        self.num_threads = max(os.cpu_count() - 2, 1)

        # Download the model the first time, so the user never has to run `ollama pull`
        self._ensure_model_downloaded()

    def _ensure_model_downloaded(self):
        # Ask the ollama server what it has, and pull our model if it's missing
        downloaded = [m.model for m in ollama.list().models]
        if not any(name.startswith(self.model) for name in downloaded):
            print(f"Downloading {self.model} (one time only, this can take a few minutes)...")
            ollama.pull(self.model)

    def _encode_image(self, image):
        # Accept either a file path or an OpenCV frame, and hand ollama raw JPEG bytes
        if isinstance(image, np.ndarray):
            ok, buffer = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError("Could not encode the camera frame as a JPEG")
            return buffer.tobytes()
        with open(image, "rb") as f:
            return f.read()

    def warm_up(self):
        # Load the model into memory now, so the first real identify() isn't minutes slow
        ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1, "num_thread": self.num_threads},
            keep_alive=self.keep_alive,
        )

    def identify(self, image):
        """Scans one image and returns a dict like {"object": "bottle"}, YOLO-style."""
        # Ask the vision model one short question, forcing a JSON answer
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "What object is in the middle of this image? Answer in 1-3 words.",
                    "images": [self._encode_image(image)],
                }
            ],
            format=self.RESPONSE_SCHEMA,
            # Capping the output tokens stops the model from rambling for minutes
            options={"num_predict": 50, "num_thread": self.num_threads},
            keep_alive=self.keep_alive,
        )

        # The schema guarantees the reply parses as {"object": ...}
        return json.loads(response.message.content)


# Simple demo: grab one frame from the laptop camera and ask what's in it
if __name__ == "__main__":
    identifier = ObjectIdentifier()
    print(f"Loading {identifier.model} into memory...")
    identifier.warm_up()

    # Capture a single frame from the local camera (video port 0)
    camera = cv2.VideoCapture(0)
    for _ in range(5):
        camera.read()  # let the auto-exposure settle
    ok, frame = camera.read()
    camera.release()
    if not ok:
        raise RuntimeError("Could not read from local camera on port 0")

    # Time the query so the user can see how long one scan takes
    start = time.perf_counter()
    result = identifier.identify(frame)
    print(f"({time.perf_counter() - start:.1f}s) {json.dumps(result)}")
