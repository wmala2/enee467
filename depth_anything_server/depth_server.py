"""
Depth Anything 3 GPU inference server — optimized for multi-rover robotics.
Features a fair, bounded FIFO queue to prevent multi-user starvation cascades.
"""

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import os
import threading
import time

# Apply PyTorch allocations before importing torch to mitigate 8GB VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import cv2
from depth_anything_3.api import DepthAnything3
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import Response
import numpy as np
import torch
import uvicorn

# Metric variant required — standard DA3 outputs relative depth, not real-world meters
MODEL_ID = "depth-anything/DA3METRIC-LARGE"


# ---------------------------------------------------------------------------
# Bounded Fair FIFO Queue Architecture
# ---------------------------------------------------------------------------
class BoundedFIFOInferenceQueue:
    """
    Thread-safe, fair FIFO queue that preserves order across 15 teams
    while capping max lag by dropping stale frames when full.
    """

    def __init__(self, maxsize=15):
        self.lock = threading.Lock()
        self.new_item_available = threading.Condition(self.lock)
        self.maxsize = maxsize
        self.queue = deque()

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

    def pop(self):
        with self.lock:
            while len(self.queue) == 0:
                self.new_item_available.wait()
            return self.queue.popleft()


# One slot per rover — sized for a full class so no rover waits more than one cycle
_work_queue = BoundedFIFOInferenceQueue(maxsize=15)
# Blocks the FastAPI startup until the GPU worker thread is actually running
_ready = threading.Event()
_model = None


def process_frame():
    """Pulls the next ordered frame from the FIFO queue and processes it on GPU."""
    global _model
    _ready.set()

    while True:
        item = _work_queue.pop()
        if item is None:  # Shutdown signal
            break

        jpeg_bytes, result, done = item

        # If another thread already tripped this event (e.g. dropped due to queue timeout), skip it
        if done.is_set():
            continue

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
                    # 378 is the model's native training resolution — resize to this before inference
                    process_res=378,
                    process_res_method="upper_bound_resize",
                )

            net_output = prediction.depth[0]
            # Scale DA3 relative output to metric meters using the tuned constants above
            metric_depth_meters = (focal_length_px * net_output) / 300.0

            result["depth"] = metric_depth_meters.astype(np.float32)
            result["shape"] = metric_depth_meters.shape
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            done.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _ready

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

    yield
    _work_queue.push(None)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "gpu": torch.cuda.get_device_name(0)}


@app.post("/depth")
async def depth_endpoint(request: Request):
    jpeg_bytes = await request.body()
    if not jpeg_bytes:
        raise HTTPException(status_code=400, detail="Empty request body.")

    result: dict = {}
    done = threading.Event()

    # Push to our fair FIFO queue structure
    _work_queue.push((jpeg_bytes, result, done))

    # run_in_executor offloads the blocking done.wait() to a thread so FastAPI's event loop stays responsive
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Depth Anything 3 Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
