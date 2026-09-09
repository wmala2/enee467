"""Camera-error PID, independent of MuJoCo and Gymnasium."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LinePID:
    """Return a wheel-speed steering correction in rad/s, positive for a left turn."""

    kp: float = 2.0
    ki: float = 0.0
    kd: float = 0.1
    limit: float = 7.0
    integral: float = 0.0
    previous: float | None = None

    def update(self, error: float | None, dt: float) -> float:
        """Integrate in seconds, suppress derivative kick, and prevent windup."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        # Missing detections reset state; the caller decides when to stop the rover.
        if error is None:
            self.integral = 0.0
            self.previous = None
            return 0.0
        derivative = 0.0 if self.previous is None else (error - self.previous) / dt
        candidate = self.integral + error * dt
        raw = self.kp * error + self.ki * candidate + self.kd * derivative
        # Freeze integration only when the current error pushes further into saturation.
        if abs(raw) <= self.limit or raw * error < 0:
            self.integral = candidate
        self.previous = error
        raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        return -float(np.clip(raw, -self.limit, self.limit))


def centroid_error(image, band=0.15, threshold=60):
    """Return normalized horizontal line error, or None when the near band is empty."""
    # Keep the detection mask available for camera diagnostics and later policy observations.
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    top = int(height * (1 - band))
    mask[top:] = np.all(image[top:] < threshold, axis=-1)
    weights = mask.sum(axis=0)
    # Reject isolated dark pixels from the caster or image noise instead of calling them tape.
    if weights.sum() < 6:
        mask[:] = False
        return None, mask
    center = (width - 1) / 2
    error = (float(np.arange(width) @ weights / weights.sum()) - center) / center
    return error, mask
