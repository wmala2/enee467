"""Compact line observations from the onboard RGB image."""

import numpy as np

from envs.pid import centroid_error


def line_features(image):
    """Return near/mid/far (x, y, visible) centroids normalized to the full image."""
    height = image.shape[0]
    features = np.zeros((3, 3), dtype=np.float32)
    # The near strip exactly matches the PID; upper strips provide curvature preview.
    for index, (start, end) in enumerate(((0.85, 1.0), (0.55, 0.85), (0.25, 0.55))):
        top, bottom = int(start * height), int(end * height)
        error, mask = centroid_error(image[top:bottom], band=1.0)
        if error is not None:
            rows, _ = np.nonzero(mask)
            features[index] = [error, 2 * (top + rows.mean()) / (height - 1) - 1, 1]
    return features
