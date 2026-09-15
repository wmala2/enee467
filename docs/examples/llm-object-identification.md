# LLM Object Identification

This example shows a small vision language model naming whatever is in the middle of a camera frame, returning a clean `{"object": "..."}` result the same way a YOLO detection would, but without needing a trained detector.

## Prerequisites

- [Ollama](https://docs.ollama.com/api/introduction) installed and running locally (`ollama serve`) — the script talks to it through the `ollama` Python client
- No manual model pull needed: `ObjectIdentifier` calls `ollama.pull("moondream")` itself the first time it runs
- A working local camera on video port 0 (the demo block opens it with `cv2.VideoCapture(0)`)
- The project's `uv sync` environment, which already includes `opencv-contrib-python` and `ollama`

## Run it

```bash
uv run LLM_hybrid/object_identifier.py
```

No flags — the script has no argument parser. It loads the `moondream` model into memory, grabs one frame from the local camera, and prints how long the scan took alongside the JSON result, e.g. `(6.2s) {"object": "water bottle"}`.

## How it works

The class asks for one fixed JSON shape back from the model every time, so the caller never has to parse free-form text.

```python
RESPONSE_SCHEMA: ClassVar[dict] = {
    "type": "object",
    "properties": {"object": {"type": "string"}},
    "required": ["object"],
}
```

On construction it picks `moondream` by default — a tiny 1.8B-parameter vision model, chosen because it is the only one fast enough to answer in seconds on a laptop CPU rather than minutes.

```python
def __init__(self, model="moondream", keep_alive="10m"):
    # moondream is a tiny (1.8B) vision model - the only one fast enough on a laptop CPU
    self.model = model
    self.keep_alive = keep_alive

    # Using most of the CPU cores roughly halves the response time
    self.num_threads = max(os.cpu_count() - 2, 1)

    # Download the model the first time, so the user never has to run `ollama pull`
    self._ensure_model_downloaded()
```

Ollama models live on disk once downloaded, but a fresh machine won't have them yet. This helper checks the server's local model list and pulls the model automatically so a student never has to run a separate `ollama pull` command by hand.

```python
def _ensure_model_downloaded(self):
    # Ask the ollama server what it has, and pull our model if it's missing
    downloaded = [m.model for m in ollama.list().models]
    if not any(name.startswith(self.model) for name in downloaded):
        print(f"Downloading {self.model} (one time only, this can take a few minutes)...")
        ollama.pull(self.model)
```

Vision models need raw image bytes, not a NumPy array or a bare file path, so this normalizes either input into a JPEG byte string before it goes over the wire to Ollama.

```python
def _encode_image(self, image):
    # Accept either a file path or an OpenCV frame, and hand ollama raw JPEG bytes
    if isinstance(image, np.ndarray):
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not encode the camera frame as a JPEG")
        return buffer.tobytes()
    with open(image, "rb") as f:
        return f.read()
```

Loading a model's weights onto the CPU the first time is slow. Calling `warm_up()` once, before the real work starts, pays that cost up front with a throwaway one-token question, so the timed `identify()` call later isn't skewed by model-loading time.

```python
def warm_up(self):
    # Load the model into memory now, so the first real identify() isn't minutes slow
    ollama.chat(
        model=self.model,
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1, "num_thread": self.num_threads},
        keep_alive=self.keep_alive,
    )
```

This is the actual identification call: one short question plus the image, with the schema forcing the reply to come back as valid, parseable JSON instead of a rambling sentence.

```python
def identify(self, image):
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
```

The demo at the bottom ties it together: warm the model up, grab one settled frame from the camera (throwing away the first few so auto-exposure has time to adjust), and time the actual scan.

```python
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
```

## See also

- [LLM_hybrid/](../../LLM_hybrid/) — the module this example depends on
- [Back to README](../../README.md)
