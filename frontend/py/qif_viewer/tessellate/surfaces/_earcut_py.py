"""Pure-Python polygon-with-holes triangulation, used only if the
`mapbox_earcut` wheel fails to install/import on a given machine.

Not as fast or as robust on pathological inputs as real earcut, but good
enough for the mostly-convex, hole-having planar faces typical of mechanical
parts: each hole is bridged into the outer ring via its nearest visible
outer-ring vertex, then the resulting simple polygon is ear-clipped.
"""
from __future__ import annotations

import numpy as np


def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_triangle(p, a, b, c) -> bool:
    d1, d2, d3 = _cross2(p, a, b), _cross2(p, b, c), _cross2(p, c, a)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def _bridge_hole(outer: list[int], hole: list[int], verts: np.ndarray) -> list[int]:
    """Splice `hole` into `outer` via the nearest outer vertex (by distance),
    duplicating the two bridge vertices, per the standard hole-bridging
    technique for ear-clipping triangulators."""
    hole_start = min(range(len(hole)), key=lambda i: verts[hole[i]][0])
    h = hole[hole_start:] + hole[:hole_start] + [hole[hole_start]]
    outer_idx = min(range(len(outer)), key=lambda i: np.linalg.norm(verts[outer[i]] - verts[h[0]]))
    return outer[: outer_idx + 1] + h + outer[outer_idx:]


def _ear_clip(ring: list[int], verts: np.ndarray) -> list[list[int]]:
    idx = list(ring)
    if len(idx) > 1 and idx[0] == idx[-1]:
        idx = idx[:-1]
    if _signed_area(verts[idx]) < 0:
        idx = idx[::-1]
    triangles: list[list[int]] = []
    guard = 0
    while len(idx) > 3 and guard < 20000:
        guard += 1
        n = len(idx)
        found = False
        for i in range(n):
            i0, i1, i2 = idx[(i - 1) % n], idx[i], idx[(i + 1) % n]
            a, b, c = verts[i0], verts[i1], verts[i2]
            if _cross2(a, b, c) <= 0:
                continue
            if any(
                _point_in_triangle(verts[idx[j]], a, b, c)
                for j in range(n)
                if idx[j] not in (i0, i1, i2)
            ):
                continue
            triangles.append([i0, i1, i2])
            del idx[i]
            found = True
            break
        if not found:
            break  # degenerate remainder; stop rather than loop forever
    if len(idx) == 3:
        triangles.append(idx)
    return triangles


def triangulate(verts2d: np.ndarray, ring_ends: np.ndarray) -> np.ndarray:
    starts = [0] + list(ring_ends[:-1])
    rings = [list(range(s, e)) for s, e in zip(starts, ring_ends)]
    outer, holes = rings[0], rings[1:]
    merged = outer
    for hole in holes:
        merged = _bridge_hole(merged, hole, verts2d)
    triangles = _ear_clip(merged, verts2d)
    if not triangles:
        return np.zeros((0,), dtype=np.uint32)
    return np.array(triangles, dtype=np.uint32).ravel()
