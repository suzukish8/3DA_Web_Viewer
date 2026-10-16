"""Nurbs23 tessellation (untrimmed full-domain grid).

Only faces with a single loop that spans the whole sampled surface (an
untrimmed parametric patch, e.g. the thin blend band around a hole rim) are
sampled over the whole (u, v) domain. Anything else raises, so the caller degrades
to the wireframe fallback exactly as before -- a general trimmed-NURBS
tessellator would need point inversion onto the surface.
"""
from __future__ import annotations

import numpy as np

from qif_viewer.model.brep import Face, NurbsSurface

from . import register
from .base import TriangleSoup
from ..nurbs import eval_surface, sample_params


@register(NurbsSurface)
def tessellate_nurbs(
    face: Face,
    surface: NurbsSurface,
    loop_polylines: list[np.ndarray],
    loop_forms: list[str],
    warnings: list[str],
) -> TriangleSoup:
    if len(loop_polylines) != 1 or len(loop_polylines[0]) < 3:
        raise ValueError("trimmed NURBS surface not supported")
    nu, nv = surface.ctrl.shape[:2]
    def per_span(order: int, n_ctrl: int) -> int:
        if order <= 2:
            return 1
        return max(1, -(-24 // max(1, n_ctrl - order + 1)))

    us = sample_params(surface.order_u, surface.knots_u, nu, per_span(surface.order_u, nu))
    vs = sample_params(surface.order_v, surface.knots_v, nv, per_span(surface.order_v, nv))
    grid = np.array([
        [eval_surface(surface.order_u, surface.order_v, surface.knots_u, surface.knots_v,
                      surface.ctrl, surface.weights, u, v) for v in vs]
        for u in us
    ])
    m, n = len(us), len(vs)
    points = grid.reshape(-1, 3)
    # Untrimmed-patch check: the face's loop must span the whole grid.
    loop = loop_polylines[0]
    tol = 1e-3 * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) + 1e-6
    if (np.abs(loop.min(axis=0) - points.min(axis=0)).max() > tol
            or np.abs(loop.max(axis=0) - points.max(axis=0)).max() > tol):
        raise ValueError("trimmed NURBS surface not supported")
    tris = []
    for i in range(m - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris.append([a, c, d])
            tris.append([a, d, b])
    triangles = np.array(tris, dtype=np.int64)
    if face.turned:
        triangles = triangles[:, [0, 2, 1]]
    return TriangleSoup(points=points, triangles=triangles)
