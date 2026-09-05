import cv2
import numpy as np

from ArUco_detector.camera_stream import CameraStream

# Camera Calibration Defaults
camera_matrix_default = np.array(
    [[372.69444659, 0.0, 321.25586034], [0.0, 370.94214518, 238.16525187], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# Distortion Coeffcients
dist_coeffs_default = np.array(
    [-0.07753969, 0.11434413, -0.00108406, 0.00236484, -0.06344637], dtype=np.float32
)

marker_length_m_default = 0.10  # 10 cm


class ArucoPoseEstimator:
    """
    Finds ArUco tags in the ESP32 camera stream and reports each tag's pose.

    The "position" [X, Y, Z] is in meters and describes where the tag is *as seen
    from the rover* (the camera is the origin): +X to the right, +Y up, +Z forward.

    Frames come from a background reader (CameraStream) so the control loop never blocks
    on the network. Use process() for the freshest available pose; use next_frame()/detect()
    when you need frames captured *after* a chosen instant (e.g. after the rover stopped),
    which is how the maze runners beat the camera's 1-2 frame lag.
    """

    def __init__(
        self,
        http_addr,
        camera_matrix=camera_matrix_default,
        dist_coeffs=dist_coeffs_default,
        marker_length_m=marker_length_m_default,
        dictionary=cv2.aruco.DICT_4X4_250,
        alpha=0.25,
        max_reprojection_error_px=6.0,
        verbose=False,
    ):
        ### Camera + system configuration ###
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length_m = marker_length_m
        self.http_addr = http_addr
        self.alpha = alpha
        # Reject a detected pose if its corners reproject more than this many pixels off (blurred/occluded)
        # 6.0 px is a realistic threshold for JPEG-compressed frames from the ESP32 camera
        self.max_reprojection_error_px = max_reprojection_error_px
        self.verbose = verbose

        ### ArUco detector setup ###
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        detector_params = cv2.aruco.DetectorParameters()
        # Refine each detected corner to sub-pixel accuracy -> steadier, more precise pose
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, detector_params)

        # CLAHE boosts local contrast before detection, helping under uneven classroom lighting
        # and compensating for JPEG compression softening the marker edges
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        ### Marker corner positions in the marker's own frame (top-left, top-right, bottom-right, bottom-left) ###
        half = self.marker_length_m / 2.0
        self._marker_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        ### Pose smoothing storage (filled by process(), in output frame) ###
        self._filtered_pos = {}
        self._filtered_yaw = {}

        ### Background camera reader: always holds the freshest frame, stamped with arrival time ###
        self.camera = CameraStream(http_addr).start()

    @staticmethod
    def rotation_matrix_to_yaw_deg(R):
        """
        Extract marker yaw angle from a rotation matrix.
        """
        yaw_rad = np.arctan2(R[1, 0], R[0, 0])
        return np.degrees(yaw_rad)

    @staticmethod
    def _to_output_frame(tvec):
        # OpenCV reports the tag in the camera frame with +Y pointing DOWN; flip it so our
        # output is the tag as seen from the rover with +X right, +Y up, +Z forward (meters).
        x, y, z = tvec.reshape(-1).tolist()
        return [x, -y, z]

    def detect(self, frame):
        """
        Raw per-frame detection + pose (no smoothing). Returns (annotated_or_None, poses),
        where poses maps marker_id -> {"position": [X,Y,Z], "yaw": deg, "distance": m}.
        """
        if frame is None:
            return None, {}

        ### Detect ArUco markers ###
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        corners, ids, _ = self.detector.detectMarkers(gray)

        poses = {}
        good_corners, good_ids = [], []  # only markers that pass the quality gate

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                ### Pose estimation (cv2.aruco.estimatePoseSingleMarkers was removed in OpenCV 4.7+) ###
                # SOLVEPNP_IPPE_SQUARE is the solver built for square fiducial markers
                ok, rvec, tvec = cv2.solvePnP(
                    self._marker_points,
                    corners[i].reshape((-1, 1, 2)),
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok:
                    continue

                ### Quality gate: reproject the corners and reject the pose if it lands too far off ###
                # A clean detection reprojects to within ~1 px; a blurred or occluded one is much worse
                projected, _ = cv2.projectPoints(
                    self._marker_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
                )
                reproj_error = float(
                    np.mean(
                        np.linalg.norm(projected.reshape(4, 2) - corners[i].reshape(4, 2), axis=1)
                    )
                )
                if reproj_error > self.max_reprojection_error_px:
                    if self.verbose:
                        # Draw rejected markers in red so the viewer makes the problem visible
                        cv2.aruco.drawDetectedMarkers(
                            frame, [corners[i]], np.array([[marker_id]]), borderColor=(0, 0, 255)
                        )
                        cv2.putText(
                            frame,
                            f"ID:{marker_id} REJECTED E:{reproj_error:.1f}px",
                            (10, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                    continue

                ### Yaw from the rotation, position back in the rover-relative output frame ###
                R, _ = cv2.Rodrigues(rvec)
                poses[int(marker_id)] = {
                    "position": self._to_output_frame(tvec),
                    "yaw": float(self.rotation_matrix_to_yaw_deg(R)),
                    "distance": float(np.linalg.norm(tvec)),
                    "reproj_error": reproj_error,
                }
                good_corners.append(corners[i])
                good_ids.append([marker_id])

                ### Optional visualization ###
                if self.verbose:
                    cv2.drawFrameAxes(
                        frame,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                        self.marker_length_m * 0.5,
                    )
                    cv2.putText(
                        frame,
                        f"ID:{marker_id} R:{poses[int(marker_id)]['distance']:.2f}m Y:{poses[int(marker_id)]['yaw']:+.1f}deg E:{reproj_error:.1f}px",
                        (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

        # Draw only quality-gate-passing markers in green (rejected ones already drawn in red above)
        if self.verbose and good_corners:
            cv2.aruco.drawDetectedMarkers(frame, good_corners, np.array(good_ids))

        return (frame if self.verbose else None), poses

    def _smooth(self, poses):
        # Exponential moving average per marker, so the continuous-drive examples get a steady pose
        for marker_id, pose in poses.items():
            position = np.array(pose["position"], dtype=float)
            yaw = pose["yaw"]

            # First sighting seeds the filter; after that we blend new readings in
            if marker_id not in self._filtered_pos:
                self._filtered_pos[marker_id] = position
                self._filtered_yaw[marker_id] = yaw

            self._filtered_pos[marker_id] = (
                self.alpha * position + (1.0 - self.alpha) * self._filtered_pos[marker_id]
            )
            self._filtered_yaw[marker_id] = (
                self.alpha * yaw + (1.0 - self.alpha) * self._filtered_yaw[marker_id]
            )

            smoothed = self._filtered_pos[marker_id]
            pose["position"] = smoothed.tolist()
            pose["yaw"] = float(self._filtered_yaw[marker_id])
            pose["distance"] = float(np.linalg.norm(smoothed))
        return poses

    def process(self):
        """
        Detect on the freshest available frame (non-blocking) and return smoothed poses.
        Returns (annotated_or_None, poses).
        """
        frame, _ = self.camera.latest()
        annotated, poses = self.detect(frame)
        return annotated, self._smooth(poses)

    def next_frame(self, since, timeout=2.0):
        """
        Wait for the next camera frame that ARRIVED after `since`, and return (frame, stamp).
        Use the returned stamp as `since` on the next call to step through genuinely fresh
        frames. Returns (None, since) on a brief timeout; raises if the camera has truly stalled.
        """
        frame, stamp = self.camera.next_after(since, timeout=timeout)
        if frame is None and self.camera.is_stale():
            raise RuntimeError(
                f"Camera stream stalled - no fresh frames from {self.http_addr}. Check the ESP32 camera."
            )
        return frame, stamp

    def reset_filter(self):
        """
        Forget the smoothing history (e.g. after the rover has moved a lot).
        """
        self._filtered_pos.clear()
        self._filtered_yaw.clear()

    def get_position(self, marker_id):
        """
        Standalone getter: last smoothed (X, Y, Z) for a marker, or None if not seen yet.
        """
        if marker_id not in self._filtered_pos:
            return None
        return self._filtered_pos[marker_id].tolist()

    def get_all_positions(self):
        """
        Returns all smoothed marker positions (+X right, +Y up, +Z forward, meters).
        """
        return {mid: pos.tolist() for mid, pos in self._filtered_pos.items()}

    def close(self):
        """
        Stop the background camera reader.
        """
        self.camera.stop()
