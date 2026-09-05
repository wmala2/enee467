# External Libraries
import os
from pathlib import Path

import cv2

# Local Files to Import
from depth_anything_3.api import DepthAnything3
import numpy as np
import torch

from YOLO_agent.YOLO_extractor import YOLOExtractor

# The locally downloaded YOLOv8n weights that ship with the YOLO_agent folder
yolo_model_path = Path(__file__).resolve().parents[2] / "YOLO_agent" / "models" / "yolov8n.pt"

print("Model Path: ", yolo_model_path)
extractor = YOLOExtractor(
    model_path=yolo_model_path,
    imgsz=960,
    verbose=False,  # Turned verbose to false to prevent terminal flooding
)
# Desired Minimum Confidence
confidence_threshold = 0.80

# 1. Initialize optimized DA3 model on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
model = DepthAnything3.from_pretrained("depth-anything/DA3Metric-Large")
device = "cuda"
model = model.to(device=device).eval()

# Apply PyTorch optimizations
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model)
torch.backends.cudnn.benchmark = True


# Helper function to clean background outliers inside the target bounding box
def calculate_filtered_distance(corners, metric_depth_map):
    """
    Isolates target distance by locating the highest density peak in the
    pixel depth histogram, eliminating widely dispersed background noise.
    """
    x1, y1, x2, y2 = map(int, corners)
    box_depths = metric_depth_map[y1:y2, x1:x2].flatten()
    if len(box_depths) == 0:
        return 0.0, None, None

    # Prune lens noise floor
    box_depths = box_depths[box_depths > 0.1]
    if len(box_depths) == 0:
        return 0.0, None, None

    # 1. Create a fine-grained histogram (5cm bins from 0 to 10 meters)
    bin_size = 0.05
    bins = np.arange(0, 10.0 + bin_size, bin_size)
    counts, bin_edges = np.histogram(box_depths, bins=bins)

    # 2. Identify the index of the bin with the highest frequency count
    max_bin_idx = np.argmax(counts)

    # 3. Define a window around that peak (e.g., +/- 15cm) to capture the object's thickness
    peak_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0
    tolerance = 0.15

    lower_bound = peak_center - tolerance
    upper_bound = peak_center + tolerance

    # 4. Filter out everything except pixels inside our high-density peak window
    filtered_pixels = box_depths[(box_depths >= lower_bound) & (box_depths <= upper_bound)]

    # Fallback to median if the slice comes up empty
    if len(filtered_pixels) == 0:
        final_dist = float(np.median(box_depths))
    else:
        final_dist = float(np.mean(filtered_pixels))

    return final_dist, box_depths, filtered_pixels


# 2. Setup OpenCV video capture (0 for webcam, or pass a video path string)
cap = cv2.VideoCapture(0)
print("Starting stream. Press 'q' on any active image window to exit.")

# Wrap loop in inference mode to disable autograd tracking overhead
with torch.inference_mode():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection on the frame
        annotated, detections = extractor.process(frame)

        # Apply PyTorch optimization: Run with Automatic Mixed Precision (FP16)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model.inference(
                [frame], process_res=378, process_res_method="upper_bound_resize"
            )

        # Return the raw metric depth map
        depth_output = prediction.depth[0]
        if isinstance(depth_output, torch.Tensor):
            depth_output = depth_output.cpu().numpy()

        # Compute mathematically precise metrics using your true OpenCV calibration
        focal_length_px = 371.818
        metric_depth_meters = (focal_length_px * depth_output) / 300.0

        # Loop through YOLO detections and overlay distance calculations
        for detection in detections:
            if detection["confidence"] > confidence_threshold:
                corners = detection["box"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, corners)

                # 1. Extract the stable filtered distance in meters from the box area
                distance_m = calculate_filtered_distance(corners, metric_depth_meters)

                # Print metrics cleanly to the terminal
                print(
                    f"Target Spotted: {detection['name']} ({detection['confidence']:.2f}) -> {distance_m:.2f} meters away"
                )

                # 2. Draw a striking custom bounding box border directly onto the frame
                # (Bright Neon Green rectangle with a thickness of 3 pixels)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # 3. Create a dark solid background banner above the box for text readability
                label_text = f"{detection['name']}: {distance_m:.2f}m"
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_w, text_h = text_size

                # Draw solid background rectangle for the text label
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_h - 15),
                    (x1 + text_w + 10, y1),
                    (0, 255, 0),
                    cv2.FILLED,
                )

                # 4. Paint the text (Black text on top of the Neon Green solid banner)
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Post-processing: Prepare depth map for presentation
        max_display_depth = 10.0  # meters
        depth_clipped = np.clip(metric_depth_meters, 0, max_display_depth)
        depth_visual = ((1.0 - (depth_clipped / max_display_depth)) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)

        # Show the processed visualization frames
        if annotated is not None:
            cv2.imshow("YOLOv8n Spatial Detection", annotated)
        cv2.imshow("Calibrated Metric Depth Map", depth_colormap)

        # Consolidated singular waitKey window handler
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
