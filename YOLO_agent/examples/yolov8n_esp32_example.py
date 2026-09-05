# External Libraries
from pathlib import Path

import cv2

# Local Files to Import
from YOLO_agent.YOLO_extractor import YOLOExtractor

### Configuration ###
# The locally downloaded YOLOv8n weights that ship with this folder
MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.pt"

# The ESP32 camera serves JPEG stills at this address (see ArUco_detector/README.md)
HTTP_ADDR = "http://192.168.50.123:80/capture"

# 10-15 Hz is a good polling speed for the onboard ESP32 camera
TARGET_FPS = 5.0


def main():
    # Create the detector once, outside the main loop (imgsz=960 finds smaller objects, ~9 Hz on
    # CPU)
    extractor = YOLOExtractor(model_path=MODEL_PATH, imgsz=640, verbose=True)

    print("YOLOv8n ESP32 camera demo - press Q in the window to quit")
    while True:
        # Grab one JPEG still from the ESP32 camera, just like the ArUco tracker does
        try:
            frame = extractor.get_frame_from_http(HTTP_ADDR)
        except Exception as error:  # noqa: BLE001 -- one bad frame should not end the demo loop
            print(f"Camera error: {error}")
            continue

        # Run object detection on the frame
        annotated, detections = extractor.process(frame)

        # Print what the network found this frame
        for detection in detections:
            print(f"{detection['name']} ({detection['confidence']:.2f}) at {detection['center']}")

        # Show the annotated image with boxes and the frame rate
        if annotated is not None:
            cv2.imshow("YOLOv8n ESP32", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Sleep only the leftover time so the loop holds the target rate
        extractor.sleep_to_fps(TARGET_FPS)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
