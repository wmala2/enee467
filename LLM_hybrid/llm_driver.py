"""Proof of concept: an LLM watches the camera and decides how the rover should drive.

Instead of a control loop built from geometry and PID math, this asks a vision LLM one simple
question every frame -- "do you see the target, and is it left, center, or right?" -- and turns
that plain-English answer straight into a wheel-speed command.
"""

# External Libraries
import json
import os
import time
from typing import ClassVar

import cv2
import ollama


class LLMDriver:
    """Uses a vision LLM's left/center/right judgment to steer the rover toward a target."""

    # Speeds (m/s) used when the LLM decides to drive - well under the rover's 0.35 max
    CRUISE_SPEED = 0.20
    SLOW_SPEED = 0.10

    # The exact JSON shape we force the model to reply with: see the target or not,
    # and where it sits in the image, so the answer can steer the rover
    RESPONSE_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "target_seen": {"type": "boolean"},
            "position": {"type": "string", "enum": ["left", "center", "right"]},
        },
        "required": ["target_seen", "position"],
    }

    def __init__(self, target_object, model="moondream", keep_alive="10m"):
        # What the rover should look for and drive toward (e.g. "water bottle")
        self.target_object = target_object
        self.model = model
        self.keep_alive = keep_alive

        # Using most of the CPU cores roughly halves the response time
        self.num_threads = max(os.cpu_count() - 2, 1)

        # Download the model the first time, so the user never has to run `ollama pull`
        downloaded = [m.model for m in ollama.list().models]
        if not any(name.startswith(self.model) for name in downloaded):
            print(f"Downloading {self.model} (one time only, this can take a few minutes)...")
            ollama.pull(self.model)

    def ask_llm_where_target_is(self, frame):
        """Shows the LLM one frame and returns {"target_seen": ..., "position": ...}."""
        # Hand ollama the frame as raw JPEG bytes
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Could not encode the camera frame as a JPEG")

        # One short, schema-locked question keeps the answer fast and machine-readable
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You are the front camera of a small rover. Look for a "
                        f"{self.target_object} "
                        "in this image. Report if you see it and whether it is on the left, "
                        "center, or right of the image."
                    ),
                    "images": [buffer.tobytes()],
                }
            ],
            format=self.RESPONSE_SCHEMA,
            options={"num_predict": 60, "num_thread": self.num_threads},
            keep_alive=self.keep_alive,
        )
        return json.loads(response.message.content)

    def get_velocity_message(self, frame):
        """Turns one frame into the usual rover velocity JSON: {"command": "m", ...}."""
        decision = self.ask_llm_where_target_is(frame)

        # No target in sight means the rover should hold still
        if not decision["target_seen"]:
            left, right = 0.0, 0.0
        # Target on the left: slow the left wheel so the rover turns left
        elif decision["position"] == "left":
            left, right = self.SLOW_SPEED, self.CRUISE_SPEED
        # Target on the right: slow the right wheel so the rover turns right
        elif decision["position"] == "right":
            left, right = self.CRUISE_SPEED, self.SLOW_SPEED
        # Target dead ahead: drive straight at it
        else:
            left, right = self.CRUISE_SPEED, self.CRUISE_SPEED

        return {"command": "m", "left_mps": left, "right_mps": right}


# Proof-of-concept demo: the LLM watches the local camera and "drives" the rover.
# Set SEND_TO_ROVER = True when the rover is actually on the network.
SEND_TO_ROVER = False

if __name__ == "__main__":
    driver = LLMDriver(target_object="water bottle")

    # Open the laptop's own camera (use driver-style HTTP frames on the real rover)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open local camera on port 0")

    print(f"LLM driving toward a '{driver.target_object}' - Ctrl-C to quit")
    try:
        while True:
            # Grab the freshest frame from the camera
            ok, frame = camera.read()
            if not ok:
                continue

            # Let the LLM decide how the rover should move (slow on a CPU - this is a PoC)
            start = time.perf_counter()
            msg = driver.get_velocity_message(frame)
            print(f"({time.perf_counter() - start:.1f}s) {json.dumps(msg)}")

            # Send the exact same JSON message the other rover code sends
            if SEND_TO_ROVER:
                from rover_control import network_interface

                network_interface.send_message(json.dumps(msg).encode("utf-8"))
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
