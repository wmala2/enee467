import time
import urllib.request

import cv2
import numpy as np

### Camera Calibration ###
# Replace these with your actual calibration results
camera_matrix = np.array(
    [[468.21508393, 0.0, 400.85688995], [0.0, 468.23693575, 294.55662856], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# Replace with actual distortion coefficients if available
dist_coeffs = np.array(
    [-1.09482798e-01, 1.22416369e-01, -7.01025426e-05, 1.05030430e-03, -2.63142037e-02],
    dtype=np.float32,
)

### ArUco Configuration ###
# Define the physical marker size in meters
marker_length_m = 0.10

# Create the ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

detector_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

### Camera Stream Configuration ###
# HTTP endpoint serving JPEG snapshots
HTTP_ADDR = "http://192.168.50.123:80/capture"

### Console Logging Configuration ###
# Limit console spam
print_interval_sec = 0.5
last_print_time = 0.0

### Frame Rate Configuration ###
# Target loop rate (Hz) used to optionally slow the loop down
frame_interval = 1.0 / 15.0

### Pose Filtering Configuration ###
# Exponential moving average smoothing factor
alpha = 0.25

# Storage for filtered poses by marker ID
filtered_tvecs = {}
filtered_yaws = {}


def get_frame(url):
    """
    Fetch and decode a JPEG image from an HTTP endpoint.
    """

    with urllib.request.urlopen(url, timeout=2.0) as response:
        data = np.frombuffer(response.read(), np.uint8)

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def rotation_matrix_to_yaw_deg(R):
    """
    Extract marker yaw angle from a rotation matrix.

    Returns:
        yaw_deg
    """

    yaw_rad = np.arctan2(R[1, 0], R[0, 0])

    return np.degrees(yaw_rad)


### Main Processing Loop ###
while True:
    # Start timing before image acquisition so network latency is included
    start_time = time.perf_counter()

    # Attempt to acquire a frame from the camera
    try:
        frame = get_frame(HTTP_ADDR)

    except Exception as e:
        print(f"Camera error: {e}")
        continue

    # Verify image decoding succeeded
    if frame is None:
        print("Decode failed")
        continue

    # Convert image to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # Process all detected markers
    if ids is not None and len(ids) > 0:
        # Draw marker outlines
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Where the marker's four corners sit in its own frame (top-left, top-right, bottom-right, bottom-left)
        half = marker_length_m / 2.0
        marker_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        # Iterate through each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            # Estimate this marker's pose (cv2.aruco.estimatePoseSingleMarkers was removed in OpenCV 4.7+)
            ok, rvec, tvec = cv2.solvePnP(
                marker_points,
                corners[i].reshape((-1, 1, 2)),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            # Match the old (1, 3) shape so the tvec[0][0] / tvec[0][2] indexing below still works
            tvec = tvec.reshape(1, 3)

            # Compute rotation matrix from Rodrigues vector
            R, _ = cv2.Rodrigues(rvec)

            # Compute marker yaw angle
            yaw_deg = rotation_matrix_to_yaw_deg(R)

            # Compute straight-line distance to marker
            distance_m = np.linalg.norm(tvec)

            # Compute bearing angle relative to camera centerline
            bearing_deg = np.degrees(np.arctan2(tvec[0][0], tvec[0][2]))

            # Initialize filter storage for newly observed markers
            if marker_id not in filtered_tvecs:
                filtered_tvecs[marker_id] = tvec.copy()
                filtered_yaws[marker_id] = yaw_deg

            # Apply exponential moving average filter to translation
            filtered_tvecs[marker_id] = alpha * tvec + (1.0 - alpha) * filtered_tvecs[marker_id]

            # Apply exponential moving average filter to yaw
            filtered_yaws[marker_id] = alpha * yaw_deg + (1.0 - alpha) * filtered_yaws[marker_id]

            # Use filtered values for display
            tvec_filtered = filtered_tvecs[marker_id]
            yaw_filtered = filtered_yaws[marker_id]

            # Recompute filtered navigation quantities
            distance_filtered = np.linalg.norm(tvec_filtered)

            bearing_filtered = np.degrees(np.arctan2(tvec_filtered[0][0], tvec_filtered[0][2]))

            # Draw marker coordinate axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length_m * 0.5)

            # Build concise navigation display
            overlay_text = (
                f"ID:{marker_id}  "
                f"R:{distance_filtered:.2f}m  "
                f"B:{bearing_filtered:+.1f}deg  "
                f"Y:{yaw_filtered:+.1f}deg"
            )

            # Overlay navigation data
            cv2.putText(
                frame,
                overlay_text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Periodically print pose information
            current_time = time.time()

            if current_time - last_print_time > print_interval_sec:
                print(
                    f"Marker {marker_id} | "
                    f"Range: {distance_filtered:.2f} m | "
                    f"Bearing: {bearing_filtered:+.1f} deg | "
                    f"Yaw: {yaw_filtered:+.1f} deg"
                )

                last_print_time = current_time

    # Measure total loop execution time
    processing_time = time.perf_counter() - start_time

    # Display frame rate
    fps_text = f"FPS: {1.0 / processing_time:.1f}" if processing_time > 0 else "FPS: --"
    cv2.putText(
        frame, fps_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )

    # Display processed image
    cv2.imshow("ArUco Pose Tracking", frame)

    # Optional frame-rate limiting
    sleep_time = frame_interval - processing_time

    if sleep_time > 0:
        time.sleep(sleep_time)

    # Exit when user presses Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

### Cleanup ###
cv2.destroyAllWindows()
