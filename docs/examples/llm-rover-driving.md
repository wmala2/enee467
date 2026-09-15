# LLM Rover Driving

This is a proof-of-concept where a vision language model watches the live camera feed, decides whether it can see a target object and which side of the frame it's on, and turns that decision straight into the same wheel-speed JSON message the rover's normal control loop sends.

## Prerequisites

- [Ollama](https://docs.ollama.com/api/introduction) installed and running locally (`ollama serve`)
- No manual model pull needed: `LLMDriver` calls `ollama.pull("moondream")` itself the first time it runs
- A working local camera on video port 0 (the demo opens it with `cv2.VideoCapture(0)`)
- The project's `uv sync` environment (`opencv-contrib-python`, `ollama`)
- Only if you flip `SEND_TO_ROVER = True` in the script: a rover reachable on the network, since that path imports `rover_control.network_interface` and actually sends the drive command

## Run it

```bash
uv run LLM_hybrid/llm_driver.py
```

No flags — there's no argument parser. By default `SEND_TO_ROVER` is `False`, so it only prints what it *would* send; watch the camera window's terminal output for lines like `(7.1s) {"command": "m", "left_mps": 0.1, "right_mps": 0.2}`. Press Ctrl-C to stop.

## How it works

The schema forces the model to answer two specific questions every frame — is the target visible, and where is it — rather than returning open-ended text that would need fragile parsing.

```python
RESPONSE_SCHEMA: ClassVar[dict] = {
    "type": "object",
    "properties": {
        "target_seen": {"type": "boolean"},
        "position": {"type": "string", "enum": ["left", "center", "right"]},
    },
    "required": ["target_seen", "position"],
}
```

Two fixed speeds keep this safe: both are well under the rover's real top speed, so even if the LLM's reasoning is a little off, the rover only ever creeps rather than surges.

```python
class LLMDriver:
    # Speeds (m/s) used when the LLM decides to drive - well under the rover's 0.35 max
    CRUISE_SPEED = 0.20
    SLOW_SPEED = 0.10
```

Setup mirrors the object identifier: remember what to look for, pick the fast `moondream` model by default, and auto-download it so nobody has to run `ollama pull` manually.

```python
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
```

This is the core perception step: hand the model one camera frame and a plain-English question, and get back a structured answer about where the target sits in the image.

```python
def ask_llm_where_target_is(self, frame):
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
```

This is the "brain" that turns perception into motion: it's the same differential-drive trick as line-following — slow one wheel relative to the other to turn toward the target, or stop if nothing's in sight.

```python
def get_velocity_message(self, frame):
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
```

The demo loop keeps grabbing fresh frames and asking the LLM what to do, forever, until you interrupt it. `SEND_TO_ROVER` is a manual safety switch: leave it `False` on a laptop with no rover attached, and only flip it once you actually intend to move a real rover.

```python
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
```

## See also

- [LLM_hybrid/](../../LLM_hybrid/) — the module this example depends on
- [Back to README](../../README.md)
