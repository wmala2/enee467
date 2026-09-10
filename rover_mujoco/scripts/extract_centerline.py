"""Extracts an ordered centerline polyline from a flat track mesh (STL).

    uv run python scripts/extract_centerline.py assets/objects/tracks/goomba_track.stl --scale 7

CAD tracks arrive as a surface: a band of triangles with an outer and an inner edge and no
notion of a path along it. The RL environments need the opposite, a centerline polyline, since
progress reward, deviation, and the success criterion all project the rover onto one. This
turns the first into the second.

Method: rasterize the band, flood the exterior to find the enclosed hole, take the Euclidean
distance from that hole, and read off the iso-contour at half the band width. That contour is
the centerline by construction, and marching squares hands it back already ordered, which a
distance-ridge or thinning approach would not.
"""

import argparse
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def load_stl_xy(path):
    """(N, 3, 2) array of triangle corners projected to the xy plane, in metres."""
    raw = path.read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    dtype = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
    tris = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    return tris["v"].reshape(-1, 3, 3)[:, :, :2].astype(np.float64)


def rasterize(tris, resolution):
    """Binary mask of the band, plus the (x, y) of pixel [0, 0] and the metres per pixel."""
    lo = tris.reshape(-1, 2).min(axis=0)
    hi = tris.reshape(-1, 2).max(axis=0)
    pad = 0.05 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    step = (hi - lo).max() / resolution
    width, height = int((hi[0] - lo[0]) / step) + 1, int((hi[1] - lo[1]) / step) + 1

    mask = np.zeros((height, width), dtype=bool)
    # Scanline fill per triangle; the meshes here are ~1k triangles so this is fast enough.
    for tri in tris:
        px = (tri - lo) / step
        x0, y0 = np.floor(px.min(axis=0)).astype(int)
        x1, y1 = np.ceil(px.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width - 1), min(y1, height - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        p = np.stack([xs + 0.5, ys + 0.5], axis=-1)
        a, b, c = px
        # Barycentric sign test.
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        u = ((b[1] - c[1]) * (p[..., 0] - c[0]) + (c[0] - b[0]) * (p[..., 1] - c[1])) / d
        v = ((c[1] - a[1]) * (p[..., 0] - c[0]) + (a[0] - c[0]) * (p[..., 1] - c[1])) / d
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        mask[y0 : y1 + 1, x0 : x1 + 1] |= inside
    return mask, lo, step


def centerline(mask, step):
    """Ordered centerline pixels, and the band's measured width in metres."""
    # The hole is whatever the band encloses: everything not reachable from the border.
    free = ~mask
    labels, _ = ndimage.label(free)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    hole = np.isin(labels, [x for x in np.unique(labels) if x and x not in border])
    if not hole.any():
        raise SystemExit("no enclosed region found; the track may not be a closed loop")

    # Band width from the band's own distance transform: its ridge sits at half the width.
    band_half = ndimage.distance_transform_edt(mask).max()
    distance = ndimage.distance_transform_edt(~hole)

    figure, axes = plt.subplots()
    contours = axes.contour(distance, levels=[band_half])
    paths = [p.vertices for p in contours.get_paths()]
    plt.close(figure)
    if not paths:
        raise SystemExit("no centerline contour found")
    return max(paths, key=len), float(band_half * 2 * step)  # ty: ignore[no-matching-overload]


def smooth(points, sigma):
    """Periodic Gaussian smoothing of a closed polyline.

    CAD track outlines routinely meet at cusps: the Goomba clover has three, where its lobes
    join, and they come out of the extraction at a 3 mm radius. No differential-drive rover
    takes a 3 mm corner at speed, and scaling the whole track until the cusps are drivable
    makes it far too large for the arena, so the corners get rounded instead. Smoothing is
    applied before resampling so the point spacing stays even afterwards."""
    if sigma <= 0:
        return points
    return np.stack(
        [ndimage.gaussian_filter1d(points[:, i], sigma, mode="wrap") for i in range(2)], axis=1
    )


def resample(points, count):
    """`count` points spaced evenly along the closed polyline."""
    pts = np.asarray(points, dtype=np.float64)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, cum[-1], count + 1)
    out = np.stack([np.interp(target, cum, pts[:, i]) for i in range(2)], axis=1)
    return out


def curvature_radii(pts):
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    area = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    lengths = (
        np.linalg.norm(b - a, axis=1)
        * np.linalg.norm(c - b, axis=1)
        * np.linalg.norm(c - a, axis=1)
    )
    return np.where(area > 1e-12, lengths / (4 * area + 1e-15), np.inf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", type=__import__("pathlib").Path)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the result by this")
    parser.add_argument("--points", type=int, default=144)
    parser.add_argument(
        "--smooth", type=float, default=0.0, help="Gaussian sigma in contour points; rounds cusps"
    )
    parser.add_argument("--resolution", type=int, default=900)
    parser.add_argument("--preview", type=__import__("pathlib").Path)
    parser.add_argument("--out", type=__import__("pathlib").Path, help="write waypoints as CSV")
    args = parser.parse_args()

    tris = load_stl_xy(args.stl)
    mask, lo, step = rasterize(tris, args.resolution)
    contour, band_width = centerline(mask, step)

    metres = contour * step + lo
    metres = (metres - metres.mean(axis=0)) * args.scale
    pts = resample(smooth(metres, args.smooth), args.points)

    radii = curvature_radii(np.vstack([pts, pts[1]]))
    extent = pts.max(axis=0) - pts.min(axis=0)
    perimeter = float(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1).sum())
    print(f"source band width : {band_width * 100:.2f} cm (unscaled)")
    print(f"scale             : {args.scale}x   smoothing sigma: {args.smooth}")
    print(f"extent            : {extent[0]:.3f} x {extent[1]:.3f} m")
    print(f"perimeter         : {perimeter:.3f} m")
    print(f"min curve radius  : {radii.min():.3f} m")
    print(f"waypoints         : {len(pts)}")

    if args.preview:
        figure, axes = plt.subplots(figsize=(6, 6))
        axes.plot(pts[:, 0], pts[:, 1], "-", linewidth=2)
        axes.plot(pts[0, 0], pts[0, 1], "o")
        axes.set_aspect("equal")
        axes.set_title(f"{args.stl.stem} centerline, {args.scale}x")
        figure.savefig(args.preview, dpi=140, bbox_inches="tight")
        print(f"preview           : {args.preview}")

    if args.out:
        header = (
            f"centerline extracted from {args.stl.name} by scripts/extract_centerline.py\n"
            f"scale {args.scale}x, smoothing sigma {args.smooth}, {len(pts)} points\n"
            f"x_m,y_m"
        )
        np.savetxt(args.out, pts, delimiter=",", fmt="%+.6f", header=header)
        print(f"wrote             : {args.out}")
    else:
        print("\nwaypoints = [")
        for x, y in pts:
            print(f"    ({x:+.5f}, {y:+.5f}),")
        print("]")


if __name__ == "__main__":
    main()
