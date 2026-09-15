#!/usr/bin/env python3
"""Turns a folder of chessboard photos into a camera calibration file.

Every camera lens bends light slightly, so a straight line in the real world doesn't land as a
perfectly straight line in the image. Calibration measures exactly how much distortion your
specific camera has, using a checkerboard of known size photographed from several angles, and
saves the result (a camera matrix + distortion coefficients) as a `.yaml` file you can hand to
`ArucoPoseEstimator`.
"""

import glob

import cv2
import numpy as np

# The checkerboard printed in pattern.png. OpenCV counts *inner* corners, not full squares, so a
# 10x7 grid of squares has 9x6 inner corners.
squares_X = 10  # Number of squares along X
squares_Y = 7  # Number of squares along Y
nX = squares_X - 1  # Number of inner corners along X
nY = squares_Y - 1  # Number of inner corners along Y
square_size = 0.025  # Size of one square's side, in meters

# Corner detection is only approximate at first; this tells OpenCV when to stop refining a
# corner's position (either 30 iterations, or once it moves less than 0.001 pixels per step).
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Build the "ground truth" 3D positions of every inner corner, in a flat grid on the Z=0 plane,
# scaled to the real square size. Every photo of the same checkerboard reuses this same array --
# what changes photo to photo is only where those corners land in the 2D image.
objp = np.zeros((nX * nY, 3), np.float32)
objp[:, :2] = np.mgrid[0:nX, 0:nY].T.reshape(-1, 2)
objp *= square_size

# One entry per photo where the checkerboard was found: the known 3D corner positions, and where
# those same corners actually appeared in that photo's pixels.
object_points = []  # 3D points in real world
image_points = []  # 2D points in image plane


def main():
    # Every .jpg in this folder is treated as one calibration photo.
    images = glob.glob("*.jpg")
    if not images:
        print(images)
        print("No images found in current folder!")
        return

    for fname in images:
        # OpenCV's corner detector works on grayscale, not color.
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to find every inner corner of the checkerboard in this photo.
        ret, corners = cv2.findChessboardCorners(gray, (nX, nY), None)

        if ret:
            # A photo where corners were found contributes one calibration sample: the true 3D
            # layout, and the sub-pixel-refined location OpenCV actually measured for it.
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            # Show what was detected so you can visually confirm it locked onto the real board.
            cv2.drawChessboardCorners(img, (nX, nY), corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if not object_points:
        print("No corners were detected. Calibration failed.")
        return

    # This is the actual calibration: given many (3D corner, 2D pixel) pairs from different
    # angles, solve for the camera matrix (focal length, optical center) and distortion
    # coefficients that best explain all of them at once.
    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    # Save the result so it can be reused without re-running calibration every time.
    fs = cv2.FileStorage("calibration_chessboard.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("K", mtx)
    fs.write("D", dist)
    fs.release()

    # Read it back immediately, just to demonstrate the load path `ArucoPoseEstimator` will use.
    fs = cv2.FileStorage("calibration_chessboard.yaml", cv2.FILE_STORAGE_READ)
    mtx = fs.getNode("K").mat()
    dist = fs.getNode("D").mat()
    fs.release()

    print("Camera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)


if __name__ == "__main__":
    main()
