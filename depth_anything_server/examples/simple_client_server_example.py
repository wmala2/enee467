# Internal Imports
import time

# External Imports
# Local Imports
from depth_anything_server.depth_client import RoverNavigationClient

# Locally Running Loop
if __name__ == "__main__":
    # Point to your optimized desktop GPU server
    CAMERA_IP = "192.168.50.123:80"
    SERVER_IP = "192.168.50.155:5000"

    # Initialize our pre-built rover client with the address of our desired workstation
    client = RoverNavigationClient(server_url="http://" + SERVER_IP, verbose=False)
    # cap = cv2.VideoCapture(0)

    print("Running navigation client telemetry. Press Ctrl+C to stop.")
    # Loop over time to read the camera frames from your ESP32-Cam
    try:
        while True:
            t0 = time.perf_counter()

            # 1. Grab image from rover
            frame = client.fetch_rover_frame(CAMERA_IP)
            # ret, frame = cap.read()
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
