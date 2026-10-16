"""Cone23 tessellation: same construction as cylindrical.py, but the ring
radius lerps from DiameterBottom/2 to DiameterTop/2 along the axis."""
from __future__ import annotations

import numpy as np

from qif_viewer.model.brep import ConeSurface, Face

from . import register
from .base import TriangleSoup
from ..curves import rodrigues_rotate, segments_for_sweep


@register(ConeSurface)
def tessellate_cone(
    face: Face,
    surface: ConeSurface,
    loop_polylines: list[np.ndarray],
    loop_forms: list[str],
    warnings: list[str],
) -> TriangleSoup:
    axis_dir = surface.axis_dir / np.linalg.norm(surface.axis_dir)
    t0, t1 = surface.domain_angle
    n = segments_for_sweep(t1 - t0)
    thetas = np.linspace(t0, t1, n + 1)
    r_bottom = surface.diameter_bottom / 2.0
    r_top = surface.diameter_top / 2.0

    # Absolute-angle convention (see tessellate/curves.py) -- rotate by the
    # raw domain angle, not relative to the domain's own start.
    radial = np.array([rodrigues_rotate(surface.dir_beg, axis_dir, th) for th in thetas])
    ring0 = surface.axis_point + r_bottom * radial
    ring1 = surface.axis_point + axis_dir * surface.length + r_top * radial
    points = np.vstack([ring0, ring1])

    m = len(thetas)
    tri_list = []
    for i in range(n):
        a, b = i, i + 1
        c, d = m + i, m + i + 1
        # Degenerate ring collapses to a point at r=0 (e.g. a cone apex);
        # a zero-area triangle there is harmless, so no special-casing needed.
        tri_list.append([a, b, d])
        tri_list.append([a, d, c])
    triangles = np.array(tri_list, dtype=np.int64)

    if face.turned:
        triangles = triangles[:, [0, 2, 1]]

    return TriangleSoup(points=points, triangles=triangles)
