"""Merges every Face's triangle soup into one pyvista.PolyData solid.

Per-face vertices are kept unwelded across face boundaries on purpose: it
keeps face boundaries visually crisp (hard-edge shading) and avoids the
wrong-tolerance-merge risk of welding nearby-but-unrelated points.

Each tessellator already applies `Face.turned` as its best-effort winding
hint, but `turned`'s exact sign convention isn't independently verifiable
from spec-reading alone, and one flipped face renders as an unlit black
patch. As an authoritative safety net, after tessellating every face we
re-orient each one's winding so its average normal points away from the
solid's overall centroid -- correct for the (typical, mostly star-shaped)
machined-part case, and a no-op for faces `turned` already got right.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from qif_viewer.model.document import QifDocument
from qif_viewer.model.brep import Face

from .geometry_utils import loop_chain_tolerance
from .loops import loop_to_polyline
from .surfaces import tessellate_face
from .surfaces.base import TriangleSoup


def _face_normal_and_centroid(points: np.ndarray, triangles: np.ndarray):
    p0, p1, p2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    tri_normals = np.cross(p1 - p0, p2 - p0)
    tri_areas = np.linalg.norm(tri_normals, axis=1)
    valid = tri_areas > 1e-12
    if not np.any(valid):
        return None, None
    # Area-weighted average normal and centroid, so small sliver triangles
    # (e.g. from a degenerate cone apex ring) don't skew the result.
    weights = tri_areas[valid]
    avg_normal = (tri_normals[valid] * weights[:, None]).sum(axis=0)
    tri_centroids = (p0[valid] + p1[valid] + p2[valid]) / 3.0
    avg_centroid = (tri_centroids * weights[:, None]).sum(axis=0) / weights.sum()
    return avg_normal, avg_centroid


def _reorient_away_from_centroid(soups: list[tuple[Face, TriangleSoup]]) -> None:
    if not soups:
        return
    all_points = np.vstack([s.points for _, s in soups if len(s.points)])
    if len(all_points) == 0:
        return
    solid_centroid = all_points.mean(axis=0)
    for _, soup in soups:
        if len(soup.triangles) == 0:
            continue
        normal, centroid = _face_normal_and_centroid(soup.points, soup.triangles)
        if normal is None:
            continue
        if np.dot(normal, centroid - solid_centroid) < 0:
            soup.triangles = soup.triangles[:, [0, 2, 1]]


def build_mesh(doc: QifDocument) -> tuple[pv.PolyData, list[tuple[int, int, str]]]:
    """Returns (mesh, cell_offsets) where cell_offsets maps
    [start_cell, end_cell) triangle ranges back to their source Face.id --
    cheap to build now, enables future click-to-select without rework."""
    tol = loop_chain_tolerance(doc)

    soups: list[tuple[Face, TriangleSoup]] = []
    for face in doc.faces.values():
        surface = doc.surfaces.get(face.surface_id)
        if surface is None:
            doc.warn(f"Face {face.id}: surface {face.surface_id} not found; skipped")
            continue

        loop_polylines: list[np.ndarray] = []
        loop_forms: list[str] = []
        for loop_id in face.loop_ids:
            loop = doc.loops.get(loop_id)
            if loop is None:
                doc.warn(f"Face {face.id}: loop {loop_id} not found; skipped")
                continue
            poly = loop_to_polyline(loop, doc, doc.warnings, tol=tol)
            if len(poly) >= 2:
                loop_polylines.append(poly)
                loop_forms.append(loop.form)

        soup = tessellate_face(face, surface, loop_polylines, loop_forms, doc.warnings)
        if soup.triangles is None or len(soup.triangles) == 0:
            doc.warn(f"Face {face.id}: produced no triangles; skipped")
            continue
        soups.append((face, soup))

    _reorient_away_from_centroid(soups)

    doc.warn(f"{len(soups)}/{len(doc.faces)} faces tessellated")

    if not soups:
        return pv.PolyData(), []

    all_points: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    cell_offsets: list[tuple[int, int, str]] = []
    point_offset = 0
    cell_offset = 0
    for face, soup in soups:
        all_points.append(soup.points)
        all_tris.append(soup.triangles + point_offset)
        n_tri = len(soup.triangles)
        all_colors.append(np.tile(np.asarray(face.color, dtype=np.float32), (n_tri, 1)))
        cell_offsets.append((cell_offset, cell_offset + n_tri, face.id))
        point_offset += len(soup.points)
        cell_offset += n_tri

    points = np.vstack(all_points)
    tris = np.vstack(all_tris)
    faces_arr = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()

    mesh = pv.PolyData(points, faces_arr)
    colors = (np.vstack(all_colors) * 255).clip(0, 255).astype(np.uint8)
    mesh.cell_data["FaceColor"] = colors

    # Compute cell normals ourselves rather than via mesh.compute_normals():
    # that filter's default `splitting=True` duplicates points on sharp
    # edges and can reorder/insert cells, which would silently break the
    # cell_offsets -> Face.id mapping above. We already know each
    # triangle's correct winding, so this is just reading it off directly.
    p0, p1, p2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    tri_normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    np.divide(tri_normals, lengths, out=tri_normals, where=lengths > 1e-12)
    mesh.cell_data["Normals"] = tri_normals

    return mesh, cell_offsets
