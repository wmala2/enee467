"""Builds a black line-track MJCF fragment from (x, y) waypoints.

The tracks are polylines of thin box geoms lying flat on the floor. `envs/line_scene.py`
calls this at scene-build time, so no track files are written to disk.
"""
