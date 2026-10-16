"""Cylinder23 tessellation.

Untrimmed patches are built directly from the analytic Sweep/Axis
parameters (not from trimming edges) -- the lateral surface of a cylinder is
developable/ruled, so exactly 2 rings (bottom, top) are geometrically exact
regardless of length.

Faces that carry loops are trimmed to them (see `_tessellate_trimmed`), so
holes cut through the cylinder wall (e.g. a cross-drilled hole in a boss)
are left open instead of being covered by the full analytic patch.
"""
from __future__ import annotations

import numpy as np

from qif_viewer.model.brep import CylinderSurface, Face

from . import register
from .base import TriangleSoup
from ._triangulate import triangulate
from ..curves import rodrigues_rotate, segments_for_sweep

_TWO_PI = 2.0 * np.pi
_STEP = _TWO_PI / 64.0  # max angular extent of a lifted triangle


def _unwrap_loop(theta: np.ndarray):
    """Continuous (unwrapped) angles along a closed loop. Returns
    (theta_unwrapped, net_turn); net_turn is the loop's total winding
    around the axis (0 for a loop that does not encircle it)."""
    d = np.diff(np.append(theta, theta[0]))
    d = (d + np.pi) % _TWO_PI - np.pi
    unwrapped = theta[0] + np.concatenate([[0.0], np.cumsum(d[:-1])])
    return unwrapped, float(d.sum())


def _area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clip_ring(ring: np.ndarray, a: float, b: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a (theta, t) ring to a <= theta <= b."""
    out = ring
    for edge, keep_ge in ((a, True), (b, False)):
        if len(out) == 0:
            break
        res = []
        n = len(out)
        for i in range(n):
            p0, p1 = out[i], out[(i + 1) % n]
            in0 = p0[0] >= edge if keep_ge else p0[0] <= edge
            in1 = p1[0] >= edge if keep_ge else p1[0] <= edge
            if in0:
                res.append(p0)
            if in0 != in1:
                f = (edge - p0[0]) / (p1[0] - p0[0])
                res.append(np.array([edge, p0[1] + f * (p1[1] - p0[1])]))
        out = np.array(res) if res else np.zeros((0, 2))
    return out


def _triangulate_slabs(outer: np.ndarray, holes: list, r: float):
    """Triangulate the trimmed region slab-by-slab along theta.

    One earcut call over a whole wide patch yields long fans that the later
    arc-cut multiplies into tens of thousands of slivers; cutting the region
    into theta slabs first gives ~2 triangles per slab for a plain patch.
    Cut lines never pass through a hole (each hole stays inside one slab).
    Returns (param_vertices, triangles) or (None, None) when the slab
    pieces do not reproduce the region's area (caller uses plain earcut).
    """
    lo, hi = float(outer[:, 0].min()), float(outer[:, 0].max())
    spans = [(float(h[:, 0].min()), float(h[:, 0].max())) for h in holes]
    bounds = [lo]
    k = int(np.floor(lo / _STEP)) + 1
    while k * _STEP < hi - 1e-9:
        c = k * _STEP
        if c > lo + 1e-9 and not any(a - 1e-9 < c < b + 1e-9 for a, b in spans):
            bounds.append(c)
        k += 1
    bounds.append(hi)

    verts_all = []
    tris_all = []
    offset = 0
    area_sum = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        piece = _clip_ring(outer, a, b)
        if len(piece) < 3 or abs(_area(piece)) < 1e-12:
            continue
        piece_holes = [h for h, (ha, hb) in zip(holes, spans) if ha >= a - 1e-9 and hb <= b + 1e-9]
        ring_list = [piece] + piece_holes
        v2 = np.vstack([np.column_stack([rg[:, 0] * r, rg[:, 1]]) for rg in ring_list])
        ends = np.cumsum([len(rg) for rg in ring_list]).astype(np.uint32)
        flat = triangulate(v2, ends)
        if len(flat) == 0:
            continue
        tri = np.asarray(flat, dtype=np.int64).reshape(-1, 3)
        verts_all.append(np.vstack(ring_list))
        tris_all.append(tri + offset)
        offset += sum(len(rg) for rg in ring_list)
        area_sum += abs(_area(piece)) - sum(abs(_area(h)) for h in piece_holes)
    if not tris_all:
        return None, None
    expected = abs(_area(outer)) - sum(abs(_area(h)) for h in holes)
    if expected <= 0 or abs(area_sum - expected) > 1e-3 * abs(_area(outer)):
        return None, None
    return np.vstack(verts_all), np.vstack(tris_all)


def _tessellate_trimmed(face, surface, axis_dir, loop_polylines, warnings):
    """Trim the cylinder to the face's own loops.

    Loops are unrolled to (theta, t), triangulated with earcut (outer ring +
    holes) and lifted back onto the surface. Returns None when the unrolled
    loops are not a well-formed planar region (caller falls back to the
    analytic full patch).
    """
    r = surface.diameter / 2.0
    d = surface.dir_beg - axis_dir * np.dot(surface.dir_beg, axis_dir)
    d = d / np.linalg.norm(d)
    e = np.cross(axis_dir, d)

    rings = []
    for pts in loop_polylines:
        if len(pts) < 3:
            return None
        rel = pts - surface.axis_point
        t = rel @ axis_dir
        radial = rel - np.outer(t, axis_dir)
        theta, net = _unwrap_loop(np.arctan2(radial @ e, radial @ d))
        if abs(net) > 1e-3:
            return None  # loop encircles the axis; not a simple 2D polygon
        rings.append(np.column_stack([theta, t]))

    outer_i = int(np.argmax([abs(_area(rg)) for rg in rings]))
    outer = rings[outer_i]
    if _area(outer) < 0:
        outer = outer[::-1]
    lo, hi = float(outer[:, 0].min()), float(outer[:, 0].max())

    holes = []
    for i, rg in enumerate(rings):
        if i == outer_i:
            continue
        # Bring the hole into the same 2*pi period as the outer loop.
        mid = 0.5 * (rg[:, 0].min() + rg[:, 0].max())
        shift = _TWO_PI * np.round((0.5 * (lo + hi) - mid) / _TWO_PI)
        rg = rg + np.array([shift, 0.0])
        if rg[:, 0].min() < lo - 1e-6 or rg[:, 0].max() > hi + 1e-6:
            warnings.append(f"Face {face.id}: cylinder hole straddles the surface seam; not trimmed")
            return None
        holes.append(rg)

    param, tris = _triangulate_slabs(outer, holes, r)
    if tris is None:
        ring_list = [outer] + holes
        verts2d = np.vstack([np.column_stack([rg[:, 0] * r, rg[:, 1]]) for rg in ring_list])
        ring_ends = np.cumsum([len(rg) for rg in ring_list]).astype(np.uint32)
        flat = triangulate(verts2d, ring_ends)
        if len(flat) == 0:
            return None
        tris = np.asarray(flat, dtype=np.int64).reshape(-1, 3)
        param = np.vstack(ring_list)  # (theta, t) per vertex

    # Cut every triangle along theta = k * _STEP so that no lifted flat
    # triangle spans more than _STEP of arc (a wide one would cut through
    # the cylinder interior). Shared edges are cut at identical points on
    # both sides, so the result stays crack-free.
    verts = [tuple(v) for v in param]
    tri_out: list[tuple[int, int, int]] = []
    for ia, ib, ic in tris:
        tp = param[[ia, ib, ic]]
        k0 = int(np.floor(tp[:, 0].min() / _STEP + 1e-9))
        k1 = int(np.ceil(tp[:, 0].max() / _STEP - 1e-9))
        if k1 - k0 <= 1:
            tri_out.append((int(ia), int(ib), int(ic)))
            continue
        # Each polygon is a list of (theta, t) tuples (convex).
        polys = [[tuple(tp[0]), tuple(tp[1]), tuple(tp[2])]]
        for k in range(k0 + 1, k1):
            cut = k * _STEP
            nxt = []
            for poly in polys:
                lo_poly, hi_poly = [], []
                for q in range(len(poly)):
                    p0, p1 = poly[q], poly[(q + 1) % len(poly)]
                    in0, in1 = p0[0] <= cut, p1[0] <= cut
                    if in0:
                        lo_poly.append(p0)
                    if not in0 or p0[0] == cut:
                        hi_poly.append(p0)
                    if (p0[0] < cut < p1[0]) or (p1[0] < cut < p0[0]):
                        f = (cut - p0[0]) / (p1[0] - p0[0])
                        x = (cut, p0[1] + f * (p1[1] - p0[1]))
                        lo_poly.append(x)
                        hi_poly.append(x)
                nxt.extend(pg for pg in (lo_poly, hi_poly) if len(pg) >= 3)
            polys = nxt
        for poly in polys:
            base = len(verts)
            verts.extend(poly)
            for q in range(1, len(poly) - 1):
                tri_out.append((base, base + q, base + q + 1))
    pts_param = [np.array(verts)]

    param_all = np.vstack(pts_param)
    th, tt = param_all[:, 0], param_all[:, 1]
    points = (
        surface.axis_point
        + np.outer(np.cos(th), d) * r
        + np.outer(np.sin(th), e) * r
        + np.outer(tt, axis_dir)
    )
    triangles = np.array(tri_out, dtype=np.int64)
    if face.turned:
        triangles = triangles[:, [0, 2, 1]]
    return TriangleSoup(points=points, triangles=triangles)


@register(CylinderSurface)
def tessellate_cylinder(
    face: Face,
    surface: CylinderSurface,
    loop_polylines: list[np.ndarray],
    loop_forms: list[str],
    warnings: list[str],
) -> TriangleSoup:
    axis_dir = surface.axis_dir / np.linalg.norm(surface.axis_dir)

    if loop_polylines:
        try:
            trimmed = _tessellate_trimmed(face, surface, axis_dir, loop_polylines, warnings)
        except Exception as exc:  # noqa: BLE001 - fall back to the untrimmed patch
            warnings.append(f"Face {face.id}: cylinder trimming failed ({exc!r}); using untrimmed patch")
            trimmed = None
        if trimmed is not None:
            return trimmed

    t0, t1 = surface.domain_angle
    n = segments_for_sweep(t1 - t0)
    thetas = np.linspace(t0, t1, n + 1)
    r = surface.diameter / 2.0

    # Absolute-angle convention (see tessellate/curves.py) -- rotate by the
    # raw domain angle, not relative to the domain's own start.
    radial = np.array([rodrigues_rotate(surface.dir_beg, axis_dir, th) for th in thetas])
    ring0 = surface.axis_point + r * radial
    ring1 = ring0 + axis_dir * surface.length
    points = np.vstack([ring0, ring1])

    m = len(thetas)
    tri_list = []
    for i in range(n):
        a, b = i, i + 1
        c, d = m + i, m + i + 1
        tri_list.append([a, b, d])
        tri_list.append([a, d, c])
    triangles = np.array(tri_list, dtype=np.int64)

    if face.turned:
        triangles = triangles[:, [0, 2, 1]]

    return TriangleSoup(points=points, triangles=triangles)
