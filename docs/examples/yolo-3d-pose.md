# YOLO 3D Pose

This example takes plain 2D object detection a step further: for every object it recognizes, it also estimates how far away that object is, giving it a real-world (X, Y, Z) position in meters from a single ordinary camera. It's the building block for any behavior where the rover needs to know not just *what* it sees but *where that thing actually is*.

## Prerequisites

- A laptop webcam available at video port 0
- The `ultralytics` package installed (see `YOLO_agent/README.md`) plus the YOLOv8n weights at `YOLO_agent/models/yolov8n.pt`
- `opencv-python`, `numpy`, `Pillow`, `torch`, and `transformers` (the pose estimator loads a Hugging Face monocular depth model through `transformers.pipeline`)
- Internet access on first run so the depth model (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`) can download into the local Hugging Face cache; after that it loads offline
- No GPU required — the depth model will use one automatically if `torch.cuda.is_available()`, otherwise it runs on CPU (slowly, hence the low target FPS below)

## Run it

```bash
uv run YOLO_agent/examples/yolov8n_pose_example.py
```

A window titled "YOLOv8n 3D Pose" shows your webcam feed with detection boxes, and the terminal prints each object's name, confidence, and X/Y/Z position in meters. Press `q` in the window to quit.

## How it works

The script is nearly identical in shape to the plain object-detection example, but it swaps in `YOLOPoseEstimator` (from `YOLO_agent/yolo_pose_estimator.py`) instead of the bare `YOLOExtractor`, and drops the target frame rate way down because depth estimation is the slow part:

```python
### Configuration ###
# The locally downloaded YOLOv8n weights that ship with this folder
MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.pt"

# The depth model is the slow part, so a couple frames per second is realistic on CPU
TARGET_FPS = 2.0
```

`YOLOPoseEstimator` is built on top of `YOLOExtractor` — it runs the same object detector internally, then adds a second neural network (a monocular depth model) that estimates distance from a single 2D image, something a regular camera can't measure directly. It pairs "what + where in the image" from YOLO with "how far away is each pixel" from the depth model, then converts that into a 3D point using the pinhole camera model — the same math a calibrated camera uses to map 2D pixels back to real-world coordinates:

```python
    # Create the pose estimator once (imgsz=960 finds smaller/farther objects, see YOLO_extractor)
    estimator = YOLOPoseEstimator(model_path=MODEL_PATH, imgsz=960, verbose=True)

    # Open the laptop's own camera (video port 0)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open local camera on port 0")
```

Grabbing a frame works exactly like the plain detection example:

```python
        # Grab one frame from the local camera
        ok, frame = camera.read()
        if not ok:
            print("Camera read failed")
            continue
```

The single `process()` call now does two neural network passes instead of one: YOLO finds the objects and boxes, then a shared depth pass estimates a meters-per-pixel map for the whole frame, and each detection's box center is looked up in that depth map and back-projected into a 3D `(X, Y, Z)` position in meters, using the same `(+X right, +Y down/up depending on convention, +Z forward)` coordinate frame the rover's ArUco tracker also reports:

```python
        # Detect objects and give each one a real-world (X, Y, Z) position in meters
        annotated, detections = estimator.process(frame)

        # Print what the network found this frame and where it is in 3D
        for detection in detections:
            x, y, z = detection["position"]
            print(
                f"{detection['name']} ({detection['confidence']:.2f}) "
                f"at X:{x:+.2f} Y:{y:+.2f} Z:{z:.2f} m"
            )
```

Displaying the annotated frame and handling quit is the same pattern as the plain detector:

```python
        # Show the annotated image with boxes and the frame rate
        if annotated is not None:
            cv2.imshow("YOLOv8n 3D Pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
```

The FPS throttle is called on `estimator.detector` rather than the estimator itself, since the underlying `YOLOExtractor` is what actually tracks frame timing — the pose estimator just wraps it:

```python
        # Sleep only the leftover time so the loop holds the target rate
        estimator.detector.sleep_to_fps(TARGET_FPS)

    camera.release()
    cv2.destroyAllWindows()
```

## See also

- [YOLO_agent/](../../YOLO_agent/) — the detector and pose estimator modules this example depends on
- [Back to README](../../README.md)
