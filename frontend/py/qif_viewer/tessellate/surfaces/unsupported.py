"""Fallback for any surface type with no registered tessellator, or one that
raised while tessellating. Renders a flat centroid-fan patch from the face's
own loop polylines -- visually approximate, but keeps the face present and
never crashes the whole load over one unusual surface.
"""
from __future__ import annotations

import numpy as np

from qif_viewer.model.brep import Face

from .base import TriangleSoup


def wireframe_fallback(face: Face, loop_polylines: list[np.ndarray]) -> TriangleSoup:
    usable = [p for p in loop_polylines if len(p) >= 3]
    if not usable:
        return TriangleSoup(points=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=np.int64))

    outer = max(usable, key=len)
    centroid = outer.mean(axis=0)
    points = np.vstack([centroid[None, :], outer])
    n = len(outer)
    triangles = np.array([[0, i + 1, ((i + 1) % n) + 1] for i in range(n)], dtype=np.int64)
    return TriangleSoup(points=points, triangles=triangles)
