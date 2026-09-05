# External Libraries
import csv
from datetime import datetime
from pathlib import Path
import time


class RunLogger:
    """
    Writes one CSV row per control tick / decision so a bench run can be diagnosed afterward.

    Every row is timestamped (`t` seconds since start) and carries the loop interval (`dt`, `hz`)
    automatically; the caller passes whatever else it has for that event via keyword arguments.
    Unfilled columns are left blank. The file is flushed every row so a crash still leaves a usable
    log. Files land in `<project>/logs/<name>_<timestamp>.csv`.
    """

    # Fixed column order (superset of everything any runner logs)
    COLUMNS = [
        "t",
        "dt",
        "hz",
        "phase",
        "event",
        "target_id",
        "seen",
        "bearing_raw",
        "bearing_smoothed",
        "X",
        "Y",
        "Z",
        "distance",
        "reproj_error",
        "frame_stamp",
        "frame_age",
        "frame_is_new",
        "distance_error",
        "heading_input",
        "pid_forward",
        "pid_turn",
        "cmd_left",
        "cmd_right",
        "saturated",
        "commanded_deg",
        "enc_left",
        "enc_right",
        "enc_left_delta",
        "enc_right_delta",
        "frames_sampled",
        "frames_with_tag",
        "lidar_left_mm",
        "lidar_center_mm",
        "lidar_right_mm",
    ]

    def __init__(self, name="run", enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return

        # logs/ lives at the project root (two levels up from this file: rover_control/ -> project)
        log_dir = Path(__file__).resolve().parents[1] / "logs"
        log_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"{name}_{stamp}.csv"

        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._file.flush()

        self._start = time.perf_counter()
        self._last = None
        print(f"Logging this run to {self.path}")

    @staticmethod
    def _round(value):
        # Keep floats readable; leave everything else (bools, strings, ints) as-is
        return round(value, 4) if isinstance(value, float) else value

    def log(self, **fields):
        # Write one row: auto-fill timing, take the rest from keyword arguments
        if not self.enabled:
            return

        now = time.perf_counter()
        row = dict.fromkeys(self.COLUMNS, "")
        for key, value in fields.items():
            if key in row and value is not None:
                row[key] = self._round(value)

        row["t"] = round(now - self._start, 4)
        if self._last is not None:
            interval = now - self._last
            row["dt"] = round(interval, 4)
            row["hz"] = round(1.0 / interval, 1) if interval > 0 else ""
        self._last = now

        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self.enabled:
            self._file.close()
