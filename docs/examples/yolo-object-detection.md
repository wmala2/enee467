# YOLO Object Detection

This example points a camera at the world and draws a live box around every object a YOLOv8n neural network recognizes, printing its name, confidence, and pixel location each frame. It's the simplest possible demo of the object detector that every other YOLO example in this repo builds on.

## Prerequisites

- A laptop webcam available at video port 0 (this walkthrough uses the local-camera script; `YOLO_agent/examples/yolov8n_esp32_example.py` is the same idea but pulls JPEG stills from the rover's ESP32 camera over HTTP instead)
- The `ultralytics` package installed (`python3 -m pip install ultralytics`, per `YOLO_agent/README.md`) so `from ultralytics import YOLO` resolves
- `opencv-python` (`cv2`) for camera capture and the annotated display window
- The YOLOv8n weights file at `YOLO_agent/models/yolov8n.pt` (downloaded per the setup instructions in `YOLO_agent/README.md`)
- A trained-on-COCO object in view if you want to see interesting detections (the model recognizes the 80 everyday COCO classes: bottle, person, chair, cup, etc.)

## Run it

```bash
uv run YOLO_agent/examples/yolov8n_local_example.py
```

A window titled "YOLOv8n Local" pops up showing your webcam feed with bounding boxes, labels, confidence scores, and an FPS counter drawn on top; the terminal prints one line per detected object each frame. Press `q` in the window to quit.

## How it works

Everything starts with a config section that points at the downloaded model weights and picks a target loop speed:

```python
### Configuration ###
# The locally downloaded YOLOv8n weights that ship with this folder
MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.pt"

# 10-15 Hz is a good loop rate for our cameras
TARGET_FPS = 15.0
```

The detector is a reusable class (`YOLOExtractor`, defined in `YOLO_agent/YOLO_extractor.py`) that wraps the actual neural network. It's created once, outside the loop, because loading the model weights from disk is slow and only needs to happen a single time:

```python
    # Create the detector once, outside the main loop (imgsz=960 finds smaller objects, ~9 Hz on
    # CPU)
    extractor = YOLOExtractor(model_path=MODEL_PATH, imgsz=960, verbose=True)

    # Open the laptop's own camera (video port 0)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open local camera on port 0")
```

`imgsz=960` controls the resolution the network scales each frame to before looking at it. A bigger number lets the network notice smaller or farther-away objects, but it costs speed — 960px runs at roughly 9 Hz on a CPU versus about 17 Hz at the default 640px, which is why the target loop rate below is set to 15 FPS rather than something more ambitious.

The main loop just grabs a frame from the webcam like any OpenCV program would:

```python
    while True:
        # Grab one frame from the local camera
        ok, frame = camera.read()
        if not ok:
            print("Camera read failed")
            continue
```

The actual detection work is a single call. Under the hood, `extractor.process()` runs the YOLO network on the frame, then converts YOLO's raw tensor output into a plain Python list of dictionaries — one per detected object — each holding the object's class name, how confident the network is (0-1), the bounding box corners in pixels, and the box's center point:

```python
        # Run object detection on the frame
        annotated, detections = extractor.process(frame)

        # Print what the network found this frame
        for detection in detections:
            print(f"{detection['name']} ({detection['confidence']:.2f}) at {detection['center']}")
```

Because `verbose=True` was passed when the extractor was created, `process()` also renders an annotated copy of the frame — the same image with boxes, labels, and an FPS readout burned into the pixels — which the script displays in an OpenCV window:

```python
        # Show the annotated image with boxes and the frame rate
        if annotated is not None:
            cv2.imshow("YOLOv8n Local", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
```

Neural network inference doesn't take a fixed amount of time, so instead of a plain `time.sleep()`, the script asks the extractor to sleep only however much time is left over in this frame's budget. That keeps the loop pinned close to `TARGET_FPS` no matter how long a given frame's detection took:

```python
        # Sleep only the leftover time so the loop holds the target rate
        extractor.sleep_to_fps(TARGET_FPS)

    camera.release()
    cv2.destroyAllWindows()
```

## See also

- [YOLO_agent/](../../YOLO_agent/) — the detector and pose estimator modules this example depends on
- [Back to README](../../README.md)
