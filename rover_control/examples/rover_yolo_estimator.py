"""
Multi-Rover Mission Control Dashboard — On-Demand Snapshot Edition

Maintains a side-by-side live video and depth view.
Press 'q' to quit the application.
Press 's' to instantly process and save a depth profile histogram to disk!
"""

# External Libraries
from pathlib import Path
import time

import cv2
import matplotlib
import numpy as np

# Force Matplotlib to use a headless backend so it never tries to create a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local Network Client
from depth_anything_server.depth_client import RoverNavigationClient

# Local Computer Vision Agent
from YOLO_agent.YOLO_extractor import YOLOExtractor

# ===========================================================================
# GLOBAL SETTINGS
# ===========================================================================
ROVER_CAMERA_IP = "192.168.50.123:80"
SERVER_IP = "192.168.50.155:5000"
CONFIDENCE_THRESH = 0.80

yolo_model_path = Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"


def calculate_filtered_distance(corners, metric_depth_map):
    """
    Standard Peak-Density Isolation: Finds the most popular depth bucket
    to isolate the foreground object from scattered background noise.
    """
    x1, y1, x2, y2 = map(int, corners)

    # Crop the metric-calibrated depth map matrix to the object boundaries
    box_depths = metric_depth_map[y1:y2, x1:x2].flatten()
    if len(box_depths) == 0:
        return 0.0, None, None

    box_depths = box_depths[box_depths > 0.1]  # Prune lens glare noise
    if len(box_depths) == 0:
        return 0.0, None, None

    # 1. Define fine-grained bucket increments (e.g., 5cm wide up to 10 meters)
    bucket_width = 0.05
    bins = np.arange(0.0, 10.0 + bucket_width, bucket_width)

    # 2. Count how many pixels land in each bucket
    counts, bin_edges = np.histogram(box_depths, bins=bins)

    # 3. Locate the index of the absolute most popular bucket (the mode)
    max_bucket_idx = np.argmax(counts)

    # Find the physical distance value at the center of that peak bucket
    peak_center = (bin_edges[max_bucket_idx] + bin_edges[max_bucket_idx + 1]) / 2.0

    # 4. Define a tolerance window (e.g., +/- 15cm) to isolate the target object cluster
    tolerance = 0.15
    lower_bound = peak_center - tolerance
    upper_bound = peak_center + tolerance

    # Slice out only the pixels that belong to this high-density peak cluster
    filtered_pixels = box_depths[(box_depths >= lower_bound) & (box_depths <= upper_bound)]

    # 5. Return the true target distance and data for the background snapshot engine
    if len(filtered_pixels) == 0:
        final_dist = float(peak_center)  # Fallback to the exact center of the peak bucket
    else:
        final_dist = float(np.mean(filtered_pixels))

    return final_dist, box_depths, filtered_pixels


def save_silent_histogram(box_depths, filtered_pixels, final_dist, obj_name):
    """Generates and saves a calibration chart completely in the background."""
    try:
        plt.figure(figsize=(8, 5))

        min_depth = np.percentile(filtered_pixels, 5)
        max_depth = np.percentile(filtered_pixels, 95)
        print("Min Depth: ", min_depth)
        print("Max Depth: ", max_depth)
        # Raw box pixels distribution (Red)
        plt.hist(
            box_depths, bins=25, alpha=0.4, color="red", label="Raw Box Data (Includes Background)"
        )
        # Isolated true target pixels (Green)
        plt.hist(
            filtered_pixels, bins=25, alpha=0.7, color="green", label="Isolated Target (Post-IQR)"
        )

        # Mean threshold indicator line
        plt.axvline(
            final_dist,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label=f"Calculated Range: {final_dist:.2f}m",
        )

        plt.title(f"On-Demand Spatial Profile: [{obj_name}]")
        plt.xlabel("Physical Distance (Meters)")
        plt.ylabel("Pixel Count Frequency")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")

        # Export disk asset
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"plots/snapshot_{obj_name}_{timestamp}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close()  # Clean up graph canvas memory
        print(f"\n[SUCCESS] Lab diagnostic chart saved to: '{output_filename}'")
    except Exception as e:
        print(f"\n[ERROR] Failed to save snapshot chart: {e}")


if __name__ == "__main__":
    extractor = YOLOExtractor(model_path=yolo_model_path, imgsz=640, verbose=False)
    client = RoverNavigationClient(server_url="http://" + SERVER_IP, verbose=False)

    print("\n====================================================")
    print("  ROVER MISSION CONTROL DASHBOARD ACTIVE")
    print("  -> Press 'q' on the dashboard view to close down.")
    print("  -> Press 's' to capture a diagnostic depth report.")
    print("====================================================")

    # Flag triggered manually by user keystroke
    trigger_snapshot_save = False

    try:
        while True:
            # 1. Fetch live frame from rover camera
            frame = client.fetch_rover_frame(ROVER_CAMERA_IP)
            if frame is None:
                time.sleep(0.5)
                continue

            h, w, c = frame.shape
            annotated, detections = extractor.process(frame)
            valid_detections = [d for d in detections if d["confidence"] > CONFIDENCE_THRESH]

            # Clear black placeholder for when the depth server is idle
            depth_colormap = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                depth_colormap,
                "SERVER IDLE: NO TARGET",
                (int(w * 0.2), int(h * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # 2. Query depth map if a target is tracked OR if an on-demand snapshot was ordered
            if len(valid_detections) > 0 or trigger_snapshot_save:
                depth_map = client.get_metric_depth(frame)

                if depth_map is not None:
                    # Construct matching visual colormap
                    max_display_depth = 10.0
                    depth_clipped = np.clip(depth_map, 0, max_display_depth)
                    depth_visual = ((1.0 - (depth_clipped / max_display_depth)) * 255).astype(
                        np.uint8
                    )
                    depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

                    # Handle metrics calculation
                    for i, detection in enumerate(valid_detections):
                        corners = detection["box"]
                        x1, y1, x2, y2 = map(int, corners)

                        # Extract distance and underlying data splits
                        distance_m, raw_data, filtered_data = calculate_filtered_distance(
                            corners, depth_map
                        )

                        # If the student hit 's' on this frame, process the first identified object
                        if trigger_snapshot_save and i == 0 and raw_data is not None:
                            save_silent_histogram(
                                raw_data, filtered_data, distance_m, detection["name"]
                            )

                        # Paint matching indicators onto both streams
                        for target_img in [annotated, depth_colormap]:
                            if target_img is None:
                                continue
                            cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                            label_text = f"{detection['name']}: {distance_m:.2f}m"
                            text_size, _ = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )

                            cv2.rectangle(
                                target_img,
                                (x1, y1 - text_size[1] - 15),
                                (x1 + text_size[0] + 10, y1),
                                (0, 128, 0),
                                cv2.FILLED,
                            )
                            cv2.putText(
                                target_img,
                                label_text,
                                (x1 + 5, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                    if trigger_snapshot_save and len(valid_detections) == 0:
                        print(
                            "\n[WARNING] Snapshot hotkey ignored: No visible objects detected to profile."
                        )

                # Reset the one-shot trigger flag immediately
                trigger_snapshot_save = False

            # 3. Handle dashboard layout assembly safely
            valid_annotated_frame = (
                annotated
                if (
                    annotated is not None
                    and isinstance(annotated, np.ndarray)
                    and annotated.ndim == 3
                )
                else frame
            )
            dashboard = np.hstack((valid_annotated_frame, depth_colormap))
            cv2.imshow("Rover Mission Control Dashboard", dashboard)

            # 4. Check for Keyboard Inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                trigger_snapshot_save = True

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nSafely closing windows and disconnecting.")
        cv2.destroyAllWindows()
