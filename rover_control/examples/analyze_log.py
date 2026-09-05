# External Libraries
from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np

# Where the runners write their logs
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def nums(rows, name, where=None):
    # Pull a numeric column, optionally filtered by a predicate on the row
    out = []
    for r in rows:
        if where and not where(r):
            continue
        value = r.get(name, "")
        if value not in ("", None):
            try:
                out.append(float(value))
            except ValueError:
                pass
    return np.array(out)


def describe(values):
    if len(values) == 0:
        return "n/a"
    return f"n={len(values)}  min={values.min():.3f}  median={np.median(values):.3f}  max={values.max():.3f}  mean={values.mean():.3f}"


def fraction_true(rows, name, where=None):
    flags = [r.get(name, "") for r in rows if (not where or where(r))]
    flags = [f for f in flags if f != ""]
    if not flags:
        return None
    return sum(1 for f in flags if f in ("True", "1")) / len(flags)


def main():
    # Use the file given on the command line, else the newest maze_*.csv in logs/
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        logs = sorted(LOG_DIR.glob("maze_*.csv"))
        if not logs:
            print(f"No logs found in {LOG_DIR}")
            return
        path = logs[-1]

    rows = load(path)
    print(f"Analyzing {path}  ({len(rows)} rows)\n")

    # What happened, at a glance
    print("Rows by phase:", dict(Counter(r["phase"] for r in rows)))
    print("Events:", dict(Counter(r["event"] for r in rows if r["event"])))

    # The continuous-drive ticks are the rows with phase 'drive' and no event text
    is_tick = lambda r: r["phase"] == "drive" and r["event"] == ""
    ticks = [r for r in rows if is_tick(r)]

    print("\n--- continuous drive ticks ---")
    print("count:                ", len(ticks))
    print("loop rate (hz):       ", describe(nums(ticks, "hz")))
    print("frame age (s):        ", describe(nums(ticks, "frame_age")))
    new = fraction_true(ticks, "frame_is_new", is_tick)
    if new is not None:
        print(f"acted on a NEW frame:  {new * 100:.0f}% of ticks  (low % = re-using stale frames)")
    print("bearing_raw (deg):    ", describe(nums(ticks, "bearing_raw")))
    print("bearing_smoothed(deg):", describe(nums(ticks, "bearing_smoothed")))

    # EMA lag: how far the smoothed bearing trails the raw one (a likely instability source)
    raw = nums(ticks, "bearing_raw")
    smooth = nums(ticks, "bearing_smoothed")
    if len(raw) == len(smooth) and len(raw) > 0:
        print(
            f"|raw - smoothed| (deg): mean={np.abs(raw - smooth).mean():.2f}  (big = EMA lag hurting heading)"
        )
    print("Z forward (m):        ", describe(nums(ticks, "Z")))
    print("pid_turn:             ", describe(nums(ticks, "pid_turn")))
    sat = fraction_true(ticks, "saturated", is_tick)
    if sat is not None:
        print(f"speed saturated:       {sat * 100:.0f}% of ticks")

    # Thrash: how often it bailed the drive to re-center
    recenters = sum(1 for r in rows if r["event"].startswith("recenter"))
    print(f"\nre-center bailouts:    {recenters}")

    # Open-loop turn accuracy: pair each commanded turn with the bearing measured right after it
    print("\n--- turn accuracy (commanded vs achieved) ---")
    pairs = []
    pending = None
    for r in rows:
        if r["commanded_deg"] != "" and r["event"] in ("turn to center", "re-aim"):
            pending = (
                float(r["commanded_deg"]),
                float(r["bearing_raw"]) if r["bearing_raw"] else None,
            )
        elif pending is not None and r["bearing_raw"] != "":
            commanded, before = pending
            after = float(r["bearing_raw"])
            if before is not None and abs(commanded) > 1e-6:
                pairs.append((before - after) / commanded)  # achieved / commanded
            pending = None
    if pairs:
        ratios = np.array(pairs)
        print(
            f"achieved/commanded turn ratio: mean={ratios.mean():.2f}  (1.0 = perfect; <1 = under-turning)"
        )
        print(f"  -> the rover turns about {ratios.mean() * 100:.0f}% of the commanded degrees")
    else:
        print("not enough turn->measurement pairs to estimate")


if __name__ == "__main__":
    main()
