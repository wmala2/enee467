"""Live side-by-side camera + depth dashboard for the rover, with on-demand distance diagnostics.

Two windows worth of information are combined into one: the raw YOLO-annotated camera feed next
to a false-color depth map. Press 'q' to quit, or 's' to save a histogram showing exactly how the
distance to the current object was computed.
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
    """Estimates one clean distance (in meters) to whatever's inside a YOLO box.

    A depth map's raw pixels inside a bounding box are noisy: some belong to the target, some
    belong to background poking through gaps around it. This finds the most common depth value
    in the box (its "peak") and averages only the pixels close to that peak, which throws out
    the background outliers. It also hands back the raw and filtered pixels so the histogram
    below can show exactly what happened.
    """
    x1, y1, x2, y2 = map(int, corners)

    # Crop the metric-calibrated depth map matrix to the object's bounding box.
    box_depths = metric_depth_map[y1:y2, x1:x2].flatten()
    if len(box_depths) == 0:
        return 0.0, None, None

    box_depths = box_depths[box_depths > 0.1]  # drop near-zero readings (lens glare/noise)
    if len(box_depths) == 0:
        return 0.0, None, None

    # Bucket every depth reading into 5 cm-wide bins out to 10 meters.
    bucket_width = 0.05
    bins = np.arange(0.0, 10.0 + bucket_width, bucket_width)
    counts, bin_edges = np.histogram(box_depths, bins=bins)

    # The bin with the most pixels in it (the mode) is almost always the target object rather
    # than background peeking around its edges.
    max_bucket_idx = np.argmax(counts)
    peak_center = (bin_edges[max_bucket_idx] + bin_edges[max_bucket_idx + 1]) / 2.0

    # Average only the pixels within 15 cm of that peak, for a cleaner final distance.
    tolerance = 0.15
    lower_bound = peak_center - tolerance
    upper_bound = peak_center + tolerance
    filtered_pixels = box_depths[(box_depths >= lower_bound) & (box_depths <= upper_bound)]

    if len(filtered_pixels) == 0:
        final_dist = float(peak_center)  # fall back to the peak bucket's center
    else:
        final_dist = float(np.mean(filtered_pixels))

    return final_dist, box_depths, filtered_pixels


def save_silent_histogram(box_depths, filtered_pixels, final_dist, obj_name):
    """Saves a chart of the raw-vs-filtered depth pixels, so you can see the peak-picking above
    actually working rather than trusting it blindly.
    """
    try:
        plt.figure(figsize=(8, 5))

        min_depth = np.percentile(filtered_pixels, 5)
        max_depth = np.percentile(filtered_pixels, 95)
        print("Min Depth: ", min_depth)
        print("Max Depth: ", max_depth)
        # Every pixel in the box, including background (red) vs. just the isolated target
        # cluster after filtering (green) -- the gap between them is what the filtering removed.
        plt.hist(
            box_depths, bins=25, alpha=0.4, color="red", label="Raw Box Data (Includes Background)"
        )
        plt.hist(
            filtered_pixels, bins=25, alpha=0.7, color="green", label="Isolated Target (Post-IQR)"
        )

        # Mark the final computed distance on the chart for reference.
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

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"plots/snapshot_{obj_name}_{timestamp}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close()  # free the figure's memory now that it's saved
        print(f"\n[SUCCESS] Lab diagnostic chart saved to: '{output_filename}'")
    except Exception as e:  # noqa: BLE001 -- a failed diagnostic plot must not lose the run it was plotting
        print(f"\n[ERROR] Failed to save snapshot chart: {e}")


if __name__ == "__main__":
    # YOLOExtractor finds objects in each frame; RoverNavigationClient asks the GPU server for a
    # depth map of the same frame.
    extractor = YOLOExtractor(model_path=yolo_model_path, imgsz=640, verbose=False)
    client = RoverNavigationClient(server_url="http://" + SERVER_IP, verbose=False)

    print("\n====================================================")
    print("  ROVER MISSION CONTROL DASHBOARD ACTIVE")
    print("  -> Press 'q' on the dashboard view to close down.")
    print("  -> Press 's' to capture a diagnostic depth report.")
    print("====================================================")

    # Set by the 's' key below; read (and cleared) once at the top of the next loop iteration.
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

            # Placeholder depth panel shown whenever we haven't computed a real one this frame.
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

            # 2. Only bother asking the GPU for a depth map if there's something to measure, or
            # the user explicitly asked for a snapshot -- depth inference isn't free.
            if len(valid_detections) > 0 or trigger_snapshot_save:
                depth_map = client.get_metric_depth(frame)

                if depth_map is not None:
                    # Turn the raw meters-per-pixel depth map into a viewable false-color image.
                    max_display_depth = 10.0
                    depth_clipped = np.clip(depth_map, 0, max_display_depth)
                    depth_visual = ((1.0 - (depth_clipped / max_display_depth)) * 255).astype(
                        np.uint8
                    )
                    depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

                    # Compute and label a distance for every confident detection this frame.
                    for i, detection in enumerate(valid_detections):
                        corners = detection["box"]
                        x1, y1, x2, y2 = map(int, corners)

                        distance_m, raw_data, filtered_data = calculate_filtered_distance(
                            corners, depth_map
                        )

                        # If the student hit 's' on this frame, save a histogram for the first
                        # detected object only.
                        if trigger_snapshot_save and i == 0 and raw_data is not None:
                            save_silent_histogram(
                                raw_data, filtered_data, distance_m, detection["name"]
                            )

                        # Draw the same box and distance label on both the camera and depth views.
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
                            "\n[WARNING] Snapshot hotkey ignored: No visible objects detected to "
                            "profile."
                        )

                # Snapshots are one-shot: reset the flag now that this frame handled it.
                trigger_snapshot_save = False

            # 3. Build the side-by-side dashboard, falling back to the raw frame if the annotated
            # one came back malformed.
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

            # 4. Handle key presses: 'q' quits, 's' arms a snapshot for the next frame that has
            # a depth map available.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                trigger_snapshot_save = True

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nSafely closing windows and disconnecting.")
        cv2.destroyAllWindows()
