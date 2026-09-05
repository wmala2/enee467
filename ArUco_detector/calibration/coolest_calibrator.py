#!/usr/bin/env python3
"""
Camera Calibration Program using a chessboard

This program performs camera calibration using a set of chessboard images.
"""

import glob

import cv2
import numpy as np

# Chessboard dimensions
squares_X = 10  # Number of squares along X
squares_Y = 7  # Number of squares along Y
nX = squares_X - 1  # Number of inner corners along X
nY = squares_Y - 1  # Number of inner corners along Y
square_size = 0.025  # Size of square in meters

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), ...
objp = np.zeros((nX * nY, 3), np.float32)
objp[:, :2] = np.mgrid[0:nX, 0:nY].T.reshape(-1, 2)
objp *= square_size

# Arrays to store points
object_points = []  # 3D points in real world
image_points = []  # 2D points in image plane


def main():
    images = glob.glob("*.jpg")
    if not images:
        print(images)
        print("No images found in current folder!")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nX, nY), None)

        if ret:
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            # Draw and display corners
            cv2.drawChessboardCorners(img, (nX, nY), corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if not object_points:
        print("No corners were detected. Calibration failed.")
        return

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    # Save calibration
    fs = cv2.FileStorage("calibration_chessboard.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("K", mtx)
    fs.write("D", dist)
    fs.release()

    # Load calibration (example)
    fs = cv2.FileStorage("calibration_chessboard.yaml", cv2.FILE_STORAGE_READ)
    mtx = fs.getNode("K").mat()
    dist = fs.getNode("D").mat()
    fs.release()

    print("Camera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)


if __name__ == "__main__":
    main()
