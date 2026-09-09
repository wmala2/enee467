"""Parametric waypoints for the black line tracks (assets/objects/tracks/). Shared by
scripts/gen_track.py (which turns these into MJCF geoms) and envs/tasks/line_follower_env.py
(which needs matching start poses/track shapes at runtime), so the two never drift apart.
"""

import math
import os

import numpy as np


def oval_waypoints(a=0.6, b=0.4, n=144):
    # Ellipse traced counter-clockwise, starting at its rightmost point
    return [(a * math.cos(t), b * math.sin(t)) for t in [2 * math.pi * i / n for i in range(n + 1)]]


def s_curve_waypoints(length=1.6, amplitude=0.3, n=144):
    # Sine-wave path running along +x, starting at the origin
    return [
        (x, amplitude * math.sin(2 * math.pi * x / length))
        for x in [length * i / n - length / 2 for i in range(n + 1)]
    ]


# Held-out shapes, used only to evaluate generalization -- never to train. Each one stresses
# something the two training tracks do not contain, so a policy that scores well here is
# following the line rather than having memorized an oval and a sine.


def circle_waypoints(r=0.45, n=144):
    """Closed loop of constant curvature. The oval's curvature swings between 0.27 m and
    1.5 m of radius, so a policy can learn a steering schedule keyed to where it is around
    the lap; a circle removes that cue without being any tighter than the oval's ends."""
    return [(r * math.cos(t), r * math.sin(t)) for t in [2 * math.pi * i / n for i in range(n + 1)]]


def figure8_waypoints(a=0.8, b=0.45, n=160):
    """Closed figure eight with a crossing at the origin and two opposite turns."""
    # Start away from the crossing so the initial branch is unambiguous.
    angles = np.linspace(math.pi / 2, 5 * math.pi / 2, n + 1)
    return [(a * math.sin(t), b * math.sin(2 * t)) for t in angles]


def wave_waypoints(length=1.8, amplitude=0.20, cycles=1.5, n=144):
    """Open path at three times the training s-curve's spatial frequency, so bends arrive
    far more often than anything seen in training and the far-band curvature preview has to
    actually be used rather than smoothed over.

    Amplitude and frequency are set together to land the tightest bend near 0.18 m of radius:
    harder than the oval's 0.27 m, but clear of the 0.081 m the rover physically cannot turn
    inside (MAX_LINEAR_VEL / MAX_ANGULAR_VEL). A first pass at 2 cycles and 0.25 m amplitude
    gave 0.085 m, which would have measured the drivetrain's limits, not the policy's."""
    return [
        (x, amplitude * math.sin(2 * math.pi * cycles * (x + length / 2) / length))
        for x in [length * i / n - length / 2 for i in range(n + 1)]
    ]


def hairpin_waypoints(straight=0.5, r=0.25, n=144):
    """Open path: straight out, 180 degrees around a tight bend, straight back. At r=0.25 m
    the bend is tighter than any curvature in training (the oval's tightest is 0.27 m), and
    is the case most likely to swing the line out of the camera's field of view."""
    pts = [(x, -r) for x in [-straight + straight * i / (n // 4) for i in range(n // 4)]]
    pts += [
        (r * math.sin(t), -r * math.cos(t))
        for t in [math.pi * i / (n // 2) for i in range(n // 2 + 1)]
    ]
    pts += [(-straight * i / (n // 4), r) for i in range(1, n // 4 + 1)]
    return pts


def goomba_waypoints():
    """Three-lobed clover track, traced from the CAD part rather than a formula.

    Unlike every other track here this one has no closed form: it comes from an OnShape part
    studio, exported as a surface mesh with no notion of a path along it. Its centerline was
    extracted once by scripts/extract_centerline.py and stored beside the mesh, so runtime
    needs neither the STL nor the rasterizer.

    Two things happened during that extraction and are baked into the stored file. The part is
    17.8 cm across, smaller than the rover, so it is scaled 7x. And the three points where its
    lobes meet are cusps, 3 mm of radius at source scale, which no differential-drive rover
    takes at speed; they are rounded until the tightest radius is 0.190 m, between the
    figure-eight's 0.155 m and the oval's 0.267 m."""
    path = os.path.join(os.path.dirname(__file__), "../assets/objects/tracks/goomba_centerline.csv")
    points = np.loadtxt(path, delimiter=",")
    return [(float(x), float(y)) for x, y in points]


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
