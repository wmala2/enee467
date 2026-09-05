# External Libraries
from pathlib import Path

import cv2

# Local Files to Import
from YOLO_agent.yolo_pose_estimator import YOLOPoseEstimator

### Configuration ###
# The locally downloaded YOLOv8n weights that ship with this folder
MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.pt"

# The depth model is the slow part, so a couple frames per second is realistic on CPU
TARGET_FPS = 2.0


def main():
    # Create the pose estimator once (imgsz=960 finds smaller/farther objects, see YOLO_extractor)
    estimator = YOLOPoseEstimator(model_path=MODEL_PATH, imgsz=960, verbose=True)

    # Open the laptop's own camera (video port 0)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open local camera on port 0")

    print("YOLOv8n 3D pose demo - press Q in the window to quit")
    while True:
        # Grab one frame from the local camera
        ok, frame = camera.read()
        if not ok:
            print("Camera read failed")
            continue

        # Detect objects and give each one a real-world (X, Y, Z) position in meters
        annotated, detections = estimator.process(frame)

        # Print what the network found this frame and where it is in 3D
        for detection in detections:
            x, y, z = detection["position"]
            print(
                f"{detection['name']} ({detection['confidence']:.2f}) "
                f"at X:{x:+.2f} Y:{y:+.2f} Z:{z:.2f} m"
            )

        # Show the annotated image with boxes and the frame rate
        if annotated is not None:
            cv2.imshow("YOLOv8n 3D Pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Sleep only the leftover time so the loop holds the target rate
        estimator.detector.sleep_to_fps(TARGET_FPS)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
