import os

import cv2
from depth_anything_3.api import DepthAnything3
import numpy as np
import torch

# 1. Initialize optimized DA3 model on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
model = DepthAnything3.from_pretrained("depth-anything/DA3Metric-Large")
device = "cuda"
model = model.to(device=device)

# Apply PyTorch optimization: convert to channels_last memory format
model = model.to(memory_format=torch.channels_last)

# Apply PyTorch optimization: fuse operations via torch.compile
model = torch.compile(model)

# Apply PyTorch optimization: enable cuDNN auto-tuner for static resolutions
torch.backends.cudnn.benchmark = True

# 2. Setup OpenCV video capture (0 for webcam, or pass a video path string)
cap = cv2.VideoCapture(0)
print("Starting stream. Press 'q' to exit.")

# Wrap loop in inference mode to disable autograd tracking overhead
with torch.inference_mode():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply PyTorch optimization: Run with Automatic Mixed Precision (FP16)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # 3. Model inference using the official DA3 API (handles resize internally via
            # process_res)
            # We pass the BGR frame directly; the API manages the array-to-tensor pipeline
            prediction = model.inference(
                [frame], process_res=378, process_res_method="upper_bound_resize"
            )

        # Return the metric depth map
        depth_output = prediction.depth[0]

        # According to a note with DA3METRIC-LARGE, we need to get theh focal length of our camera
        focal_length_px = frame.shape[1] * 0.85
        metric_depth_meters = (focal_length_px * depth_output) / 300.0
        max_display_depth = 10.0  # meters
        depth_clipped = np.clip(metric_depth_meters, 0, max_display_depth)

        # Convert PyTorch GPU tensor to NumPy CPU array
        if isinstance(depth_output, torch.Tensor):
            depth_output = depth_output.cpu().numpy()

        # 4. Post-processing: Normalize depth map to 0-255 range for visualization
        depth_normalized = cv2.normalize(
            depth_output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Colorize the depth map using an OpenCV colormap
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

        # Render both original and depth frames side-by-side
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Depth-Anything-V3 Live Feed", depth_colormap)

        # 5. Return some valid depth
        distance_in_meters = np.median(depth_output[0, 0])
        print(f"Object is exactly {distance_in_meters:.2f} meters away!")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
