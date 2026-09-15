# YOLO Agent

## What this is

This folder runs YOLO object detection on the rover's cameras. YOLO (**Y**ou **O**nly **L**ook
**O**nce, see the [project page](https://docs.ultralytics.com/#where-to-start)) is a neural
network that looks at one image and draws a box around every object it recognizes, labeled with
what it thinks the object is and how confident it is. It was trained on COCO (Common Objects in
Context), a dataset of millions of hand-labeled everyday objects, which is why it already knows
words like "person," "bottle," and "chair" without you training anything yourself.

## Set up
You should've done this in the main setup already: `ultralytics` is one of the workspace's base
dependencies in the root `pyproject.toml`, so a plain `uv sync` from the repository root installs
it. To double check:
```bash
uv run python -c "import ultralytics; print(ultralytics.__version__)"
```

## Step 1: Get a model file

YOLO needs a trained weights file (`.pt`) before it can detect anything. There are many
[YOLO models](https://huggingface.co/Ultralytics/models) to choose from, but we recommend
[YOLOv8n](https://huggingface.co/Ultralytics/YOLOv8#models) — the smallest one, which is why it
runs fast enough for a low-power laptop and a cheap camera. Download the `.pt` file
[here](https://huggingface.co/Ultralytics/YOLOv8/tree/main) and save it as
`YOLO_agent/models/yolov8n.pt` (the examples below expect it at that path).

## Step 2: Run the local camera example

```bash
uv run YOLO_agent/examples/yolov8n_local_example.py
```

This opens your laptop's own webcam, runs YOLOv8n on every frame, and pops up a window with a box
drawn around each object it recognizes, labeled with its name and confidence:

<!-- TODO: screenshot needed -- an annotated detection frame from yolov8n_local_example.py,
     showing a box, class label, and confidence score around a recognized object -->

All of the detection logic lives in the [`YOLOExtractor`](YOLO_extractor.py) class — the example
script just wires it up to a camera. You can import `YOLOExtractor` the same way in your own
scripts (the ArUco follower-style control loops in `rover_control/examples/` do exactly this).

## Example Executables

| Executable | Purpose |
|------------|---------|
| [YOLO Local Camera](examples/yolov8n_local_example.py) | Runs YOLOv8n object detection live on your laptop's own camera (video port 0). |
| [YOLO ESP32 Camera](examples/yolov8n_esp32_example.py) | Runs YOLOv8n object detection on JPEG stills polled from the rover's ESP32 camera. |
| [YOLO 3D Pose](examples/yolov8n_pose_example.py) | Adds a real-world (X, Y, Z) position in meters to every detection using a monocular depth model. |

## GPU Depth Server
For real-time metric depth on the rover, offload inference to a desktop GPU via the depth server. See [depth_anything_server/](../depth_anything_server/) for setup, the client API, and examples.

## How YOLO actually works, briefly

A CNN (Convolutional Neural Network) like YOLO detects objects by sliding small filters (kernels)
across the image, looking for patterns like edges and textures, then combining those patterns
through many layers until the network can say "this group of pixels looks like a bottle." YOLO's
particular trick — the "you only look once" part — is doing this in a single pass over the whole
image, which is what makes it fast enough to run live on a camera feed instead of taking seconds
per frame.

