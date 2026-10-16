"""Plane23 tessellation: 2D projection + earcut-with-holes.

Outer-vs-inner is cross-checked geometrically (largest 2D loop area wins)
rather than trusting `Loop.form` alone -- a mislabeled `form` would
otherwise silently produce an inverted hole.
"""
from __future__ import annotations

import numpy as np

from qif_viewer.model.brep import Face, PlaneSurface

from . import register
from .base import TriangleSoup
from ._triangulate import triangulate


def _project_to_2d(points: np.ndarray, origin: np.ndarray, dir_u: np.ndarray, dir_v: np.ndarray) -> np.ndarray:
    rel = points - origin
    return np.column_stack([rel @ dir_u, rel @ dir_v])


def _signed_area(poly2d: np.ndarray) -> float:
    x, y = poly2d[:, 0], poly2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@register(PlaneSurface)
def tessellate_plane(
    face: Face,
    surface: PlaneSurface,
    loop_polylines: list[np.ndarray],
    loop_forms: list[str],
    warnings: list[str],
) -> TriangleSoup:
    dir_u = surface.dir_u / np.linalg.norm(surface.dir_u)
    dir_v = surface.dir_v / np.linalg.norm(surface.dir_v)

    rings_2d: list[np.ndarray] = []
    rings_3d: list[np.ndarray] = []
    areas: list[float] = []
    for pts3d in loop_polylines:
        if len(pts3d) < 3:
            continue
        pts2d = _project_to_2d(pts3d, surface.origin, dir_u, dir_v)
        rings_2d.append(pts2d)
        rings_3d.append(pts3d)
        areas.append(abs(_signed_area(pts2d)))

    if not rings_2d:
        warnings.append(f"Face {face.id}: no usable (>=3 point) loops; empty mesh")
        return TriangleSoup(points=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=np.int64))

    outer_idx = int(np.argmax(areas))
    declared_outer_idx = next((i for i, f in enumerate(loop_forms) if i < len(loop_forms) and f == "OUTER"), None)
    if declared_outer_idx is not None and declared_outer_idx != outer_idx and len(rings_2d) > 1:
        warnings.append(
            f"Face {face.id}: Loop.form OUTER (loop index {declared_outer_idx}) disagrees with "
            f"largest-2D-area ranking (index {outer_idx}); using area ranking"
        )

    ordering = [outer_idx] + [i for i in range(len(rings_2d)) if i != outer_idx]

    outer_2d, outer_3d = rings_2d[ordering[0]], rings_3d[ordering[0]]
    if _signed_area(outer_2d) < 0:
        outer_2d, outer_3d = outer_2d[::-1], outer_3d[::-1]

    ring_2d_list = [outer_2d] + [rings_2d[i] for i in ordering[1:]]
    ring_3d_list = [outer_3d] + [rings_3d[i] for i in ordering[1:]]

    verts2d = np.vstack(ring_2d_list)
    verts3d = np.vstack(ring_3d_list)
    ring_ends = np.cumsum([len(r) for r in ring_2d_list]).astype(np.uint32)

    flat_tris = triangulate(verts2d, ring_ends)
    if len(flat_tris) == 0:
        warnings.append(f"Face {face.id}: triangulation produced no triangles")
        return TriangleSoup(points=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=np.int64))
    triangles = np.asarray(flat_tris, dtype=np.int64).reshape(-1, 3)

    desired_normal = np.cross(dir_u, dir_v)
    if face.turned:
        desired_normal = -desired_normal
        triangles = triangles[:, [0, 2, 1]]

    return TriangleSoup(points=verts3d, triangles=triangles, normal_hint=desired_normal)
