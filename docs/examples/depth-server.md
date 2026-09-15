# Depth Server

This turns a CUDA GPU desktop into a shared depth-estimation service: rovers on the same network POST a JPEG frame to it and get back a metric depth map (distance in meters for every pixel), so multiple laptops can share one expensive GPU instead of each needing their own.

## Prerequisites

- A CUDA-capable machine with at least 8 GB VRAM — the server raises `RuntimeError("No CUDA GPU detected.")` at startup if `torch.cuda.is_available()` is false
- `uv sync --extra cu121` on that machine, for a CUDA-enabled `torch`/`torchvision`
- The `depth_anything_3` package importable (`from depth_anything_3.api import DepthAnything3`) and network access to download the `depth-anything/DA3METRIC-LARGE` weights from Hugging Face the first time it runs — see `depth_anything_server/README.md` for model options
- `fastapi` and `uvicorn` installed to serve the HTTP API
- To run the client example too: the `depth_anything_server` package (installed as part of this workspace) and, if you're pointing at a real rover camera, the rover's IP reachable on the network

## Run it

```bash
uv run depth_anything_server/depth_server.py
```

Optional flags come from its `argparse` block — `--host` (default `0.0.0.0`) and `--port` (default `5000`, must be an int), e.g. `uv run depth_anything_server/depth_server.py --port 6000`. On startup you'll see `Loading depth-anything/DA3METRIC-LARGE onto GPU...` followed by a load time, then `Depth server ready — Fair Bounded FIFO Engine Active`; it then sits waiting for `POST /depth` requests until you stop it.

## How it works

### The queue: why frames wait in line fairly

With up to 15 rovers sharing one GPU, naive locking would let one chatty rover starve the rest. This class is a thread-safe FIFO: requests are served in the order they arrive, and it has a hard size cap so the queue can never grow into unbounded latency.

```python
class BoundedFIFOInferenceQueue:
    def __init__(self, maxsize=15):
        self.lock = threading.Lock()
        self.new_item_available = threading.Condition(self.lock)
        self.maxsize = maxsize
        self.queue = deque()
```

When the queue is already full and a new frame shows up, the *oldest* queued frame is dropped rather than the newest — a depth reading that's several frames stale is nearly useless anyway, so it's better to free the slot for the fresher request and let that stale client's request return promptly instead of hanging.

```python
def push(self, item):
    with self.lock:
        # If the queue is full, drop the OLDEST frame in the queue (at the left)
        # to prevent stale latency pipelines while making room for the new frame.
        if len(self.queue) >= self.maxsize:
            try:
                _, _, old_done_event = self.queue.popleft()
                old_done_event.set()  # Release the starved client request smoothly
            except IndexError:
                pass

        # Append new request to the right side of the queue (FIFO order)
        self.queue.append(item)
        self.new_item_available.notify()
```

### The GPU worker thread

FastAPI handles many HTTP requests concurrently, but there's only one GPU, so all the actual model inference happens serially on a single dedicated background thread that just keeps pulling the next item off the queue.

```python
def process_frame():
    """Pulls the next ordered frame from the FIFO queue and processes it on GPU."""
    _ready.set()

    while True:
        item = _work_queue.pop()
        if item is None:  # Shutdown signal
            break

        jpeg_bytes, result, done = item

        # If another thread already tripped this event (e.g. dropped due to queue timeout), skip it
        if done.is_set():
            continue
```

The model itself outputs *relative* depth (which pixels are farther than others), not real-world meters. These two focal-length-like constants were tuned by hand for this specific camera and model, and are what convert the raw network output into actual meters.

```python
        try:
            buf = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode JPEG payload.")

            # Empirically tuned scale constants for this camera + DA3 model combination;
            # the standard calibration matrix values gave incorrect metric estimates in practice
            fx = 152.6944
            fy = 150.9421
            focal_length_px = (fx + fy) / 2.0

            # fp16 halves VRAM usage with no meaningful accuracy loss for depth estimation
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = _model.inference(
                    [frame],
                    # 378 is the model's native training resolution — resize to this before
                    # inference
                    process_res=378,
                    process_res_method="upper_bound_resize",
                )

            net_output = prediction.depth[0]
            # Scale DA3 relative output to metric meters using the tuned constants above
            metric_depth_meters = (focal_length_px * net_output) / 300.0

            result["depth"] = metric_depth_meters.astype(np.float32)
            result["shape"] = metric_depth_meters.shape
        except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as an error field; the worker thread must survive
            result["error"] = str(exc)
        finally:
            done.set()
```

### Startup: loading the model once, before any requests are served

FastAPI's `lifespan` context runs once when the server boots and once when it shuts down. This is where the (slow) model load happens, and where the background GPU worker thread gets started — `_ready.wait()` blocks the server from reporting "ready" until that worker thread has actually confirmed it's running.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    print(f"Loading {MODEL_ID} onto GPU...")
    t0 = time.perf_counter()

    _model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()
    # channels_last layout matches how convolutions are stored on GPU — faster inference
    _model = _model.to(memory_format=torch.channels_last)
    # lets cuDNN auto-tune the fastest kernel for our fixed input resolution
    torch.backends.cudnn.benchmark = True

    print(f"Model loaded natively in {time.perf_counter() - t0:.1f}s")

    worker_thread = threading.Thread(target=process_frame, daemon=True)
    worker_thread.start()
    _ready.wait()
    print("Depth server ready — Fair Bounded FIFO Engine Active")

    try:
        yield
    finally:
        # Sentinel must reach the worker even if the app raises, or the thread never exits.
        _work_queue.push(None)
```

### The `/depth` endpoint: bridging async HTTP and the sync GPU thread

This is the main API contract: a client POSTs raw JPEG bytes and gets raw float32 depth bytes back, with the map's height and width tucked into response headers instead of a JSON body (which keeps the response small and avoids base64-encoding a whole array). The endpoint pushes the work onto the FIFO queue, then waits for the worker thread to signal that this specific request is done.

```python
@app.post("/depth")
async def depth_endpoint(request: Request):
    jpeg_bytes = await request.body()
    if not jpeg_bytes:
        raise HTTPException(status_code=400, detail="Empty request body.")

    result: dict = {}
    done = threading.Event()

    # Push to our fair FIFO queue structure
    _work_queue.push((jpeg_bytes, result, done))

    # run_in_executor offloads the blocking done.wait() to a thread so FastAPI's event loop stays
    # responsive
    await asyncio.get_event_loop().run_in_executor(None, done.wait, 5.0)

    if not done.is_set():
        raise HTTPException(status_code=504, detail="Inference timed out waiting in FIFO queue.")

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    if "depth" not in result:
        raise HTTPException(
            status_code=408, detail="Frame dropped due to queue congestion management."
        )

    depth: np.ndarray = result["depth"]
    h, w = result["shape"]

    return Response(
        content=depth.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Depth-Height": str(h), "X-Depth-Width": str(w)},
    )
```

### Command-line entry point

The bottom of the file is what actually runs when you execute the script: parse `--host`/`--port`, then hand the FastAPI `app` to `uvicorn` to serve it.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Depth Anything 3 Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
```

## How you'd call it: the client example

`depth_anything_server/examples/simple_client_server_example.py` shows the minimal loop a rover laptop runs: fetch a frame from the rover's camera, POST it to the depth server via `RoverNavigationClient`, and read off the distance at one pixel. It hardcodes the rover's camera IP and the desktop server's IP as constants you'd edit for your own network:

```python
if __name__ == "__main__":
    # Point to your optimized desktop GPU server
    CAMERA_IP = "192.168.50.123:80"
    SERVER_IP = "192.168.50.155:5000"

    # Initialize our pre-built rover client with the address of our desired workstation
    client = RoverNavigationClient(server_url="http://" + SERVER_IP, verbose=False)

    print("Running navigation client telemetry. Press Ctrl+C to stop.")
    try:
        while True:
            t0 = time.perf_counter()

            # 1. Grab image from rover
            frame = client.fetch_rover_frame(CAMERA_IP)
            if frame is None:
                time.sleep(1)
                continue

            # 2. Get true metric depth map from server
            depth_map = client.get_metric_depth(frame)
            if depth_map is None:
                continue

            # 3. [READY FOR YOLO] - Placeholder logic for object distance lookup
            # If YOLO found an object at center pixel (320, 240):
            center_distance = depth_map[240, 320]

            print(
                f"Loop latency: {(time.perf_counter() - t0) * 1000:.1f}ms | Center Target: "
                f"{center_distance:.2f} meters"
            )

            # Regulate pacing to keep network traffic balanced across all 15 rovers
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping client loop.")
```

Run it once the server is up and the IPs above match your network:

```bash
uv run depth_anything_server/examples/simple_client_server_example.py
```

## See also

- [depth_anything_server/](../../depth_anything_server/) — the module this example depends on
- [Back to README](../../README.md)
