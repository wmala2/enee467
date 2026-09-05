"""Parametric waypoints for the black line tracks (assets/objects/tracks/). Shared by
scripts/gen_track.py (which turns these into MJCF geoms) and envs/tasks/line_follower_env.py
(which needs matching start poses/track shapes at runtime), so the two never drift apart.
"""

import math

import numpy as np


def oval_waypoints(a=0.6, b=0.4, n=48):
    # Ellipse traced counter-clockwise, starting at its rightmost point
    return [(a * math.cos(t), b * math.sin(t)) for t in [2 * math.pi * i / n for i in range(n + 1)]]


def s_curve_waypoints(length=1.6, amplitude=0.3, n=48):
    # Sine-wave path running along +x, starting at the origin
    return [
        (x, amplitude * math.sin(2 * math.pi * x / length))
        for x in [length * i / n - length / 2 for i in range(n + 1)]
    ]


def path_length_table(waypoints):
    """Cumulative arc length at each waypoint (parallel array to `waypoints`), and
    whether the path is closed (a lap, like the oval) vs. open (a single traversal
    start-to-end, like the s-curve) — detected from whether the first and last
    waypoints coincide, rather than needing a separate flag per track."""
    pts = np.asarray(waypoints, dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    closed = np.linalg.norm(pts[0] - pts[-1]) < 1e-6
    return cumlen, closed


def project_arc_length(waypoints, cumlen, point, near_segment=None, window=8, closed=False):
    """Arc-length coordinate of `point`'s nearest projection onto the polyline through
    `waypoints` (whose cumulative lengths are `cumlen`, from path_length_table), and the
    index of the segment it landed on. Used to turn ground-truth (x, y) position into
    forward-progress-along-the-track reward.

    `near_segment`, when given, restricts the search to segments within `window` of it
    instead of scanning the whole path — needed because a *global* nearest-point search
    is unstable the moment the rover drifts even a bit off-track: a point can end up
    spatially closer to a distant, path-distance-unrelated stretch (e.g. snapping back to
    the start) than to the stretch it was actually just following, producing a huge
    spurious jump in "progress" for a tiny real displacement. A local window keeps
    progress continuous; pass None only for the one-off initial fix at reset(). For a
    closed (looping) track, `closed=True` wraps that window around the seam (segment
    n-1 back to segment 0) instead of clipping there, so lap N+1 is reachable at all."""
    pts = np.asarray(waypoints, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    n_segments = len(pts) - 1
    if near_segment is None:
        segment_range = range(n_segments)
    elif closed:
        segment_range = [(near_segment + off) % n_segments for off in range(-window, window + 1)]
    else:
        segment_range = range(
            max(0, near_segment - window), min(n_segments, near_segment + window + 1)
        )

    best_s, best_d2, best_i = 0.0, np.inf, (near_segment if near_segment is not None else 0)
    for i in segment_range:
        a, b = pts[i], pts[i + 1]
        ab = b - a
        seg_len2 = ab.dot(ab)
        t = 0.0 if seg_len2 < 1e-12 else np.clip((p - a).dot(ab) / seg_len2, 0.0, 1.0)
        d2 = np.sum((p - (a + t * ab)) ** 2)
        if d2 < best_d2:
            best_d2 = d2
            best_s = cumlen[i] + t * (cumlen[i + 1] - cumlen[i])
            best_i = i
    return best_s, best_i
