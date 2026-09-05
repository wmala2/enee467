# External Libraries
from pathlib import Path

import cv2

# Local Files to Import
from YOLO_agent.YOLO_extractor import YOLOExtractor

### Configuration ###
# The locally downloaded YOLOv8n weights that ship with this folder
MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.pt"

# 10-15 Hz is a good loop rate for our cameras
TARGET_FPS = 15.0


def main():
    # Create the detector once, outside the main loop (imgsz=960 finds smaller objects, ~9 Hz on CPU)
    extractor = YOLOExtractor(model_path=MODEL_PATH, imgsz=960, verbose=True)

    # Open the laptop's own camera (video port 0)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open local camera on port 0")

    print("YOLOv8n local camera demo - press Q in the window to quit")
    while True:
        # Grab one frame from the local camera
        ok, frame = camera.read()
        if not ok:
            print("Camera read failed")
            continue

        # Run object detection on the frame
        annotated, detections = extractor.process(frame)

        # Print what the network found this frame
        for detection in detections:
            print(f"{detection['name']} ({detection['confidence']:.2f}) at {detection['center']}")

        # Show the annotated image with boxes and the frame rate
        if annotated is not None:
            cv2.imshow("YOLOv8n Local", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Sleep only the leftover time so the loop holds the target rate
        extractor.sleep_to_fps(TARGET_FPS)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
