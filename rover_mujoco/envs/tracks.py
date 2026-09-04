"""Parametric waypoints for the black line tracks (assets/objects/tracks/). Shared by
scripts/gen_track.py (which turns these into MJCF geoms) and envs/tasks/line_follower_env.py
(which needs matching start poses/track shapes at runtime), so the two never drift apart.
"""

import math


def oval_waypoints(a=0.6, b=0.4, n=48):
    # Ellipse traced counter-clockwise, starting at its rightmost point
    return [(a * math.cos(t), b * math.sin(t)) for t in
            [2 * math.pi * i / n for i in range(n + 1)]]


def s_curve_waypoints(length=1.6, amplitude=0.3, n=48):
    # Sine-wave path running along +x, starting at the origin
    return [(x, amplitude * math.sin(2 * math.pi * x / length))
            for x in [length * i / n - length / 2 for i in range(n + 1)]]
