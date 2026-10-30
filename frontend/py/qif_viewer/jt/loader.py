"""Open a JT file on its own: tessellated geometry from the Shape LOD
segments plus the PMI overlay, packaged as the same ``StepLoadResult`` the
STEP path produces so the viewer needs no JT-specific display code.

JT carries B-Rep *face groups* on its triangles (spec "B-Rep Face Group
Associations"); each group becomes one pickable face here, and the borders
between groups are drawn as the model edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qif_viewer.model.brep import Edge, Face, OcctSurface
from qif_viewer.model.document import QifDocument

from .overlay import overlay_jt_pmi
from .scene import JtScene, load_jt_scene

_DEFAULT_MODEL_COLOR = (128 / 255, 128 / 255, 128 / 255)
_EDGE_MIN_ANGLE_DEG = 20.0  # group borders flatter than this are tangent joins, not edges


@dataclass
class JtGeometry:
    """Decoded JT model, numpy only (no VTK / OCP / pyvista), so the browser
    build can use it too."""
    doc: QifDocument
    points: np.ndarray
    triangles: np.ndarray
    cell_offsets: list[tuple[int, int, str]]
    edge_ids: list[str] = field(default_factory=list)
    edge_polylines: list[np.ndarray] = field(default_factory=list)


class PointsFaces:
    """Minimal mesh holder (``points`` + VTK-style flat ``faces``) for code that
    only needs those two arrays."""

    def __init__(self, points: np.ndarray, triangles: np.ndarray):
        self.points = points
        self.faces = np.hstack([np.full((len(triangles), 1), 3, dtype=np.int64), triangles]).ravel()


def build_jt_geometry(path: str) -> JtGeometry:
    scene = load_jt_scene(path)
    if len(scene.triangles) == 0:
        raise ValueError(
            "No tessellated geometry could be decoded from this JT file"
            + (f" ({scene.warnings[0]})" if scene.warnings else "")
        )
    doc = QifDocument(source_path=path)
    cell_offsets = _build_faces(scene, doc)
    edge_ids, polylines = _build_edges(scene, doc)
    doc.warn(f"JT import: {len(scene.face_groups)} faces, {len(edge_ids)} edges, {len(scene.triangles)} triangles")
    for w in scene.warnings:
        doc.warn(f"JT geometry: {w}")
    return JtGeometry(doc, scene.points, scene.triangles, cell_offsets, edge_ids, polylines)


def apply_jt_pmi(geom: JtGeometry, mesh, path: str) -> None:
    """Install the JT's PMI into ``geom.doc`` (``mesh``: anything with
    ``points`` / ``faces``; a VTK dataset makes the snapping faster)."""
    doc = geom.doc
    try:
        stats = overlay_jt_pmi(doc, mesh, geom.cell_offsets, path)
        doc.warn(
            f"JT PMI: {stats.characteristics} geometric tolerances, {stats.dimensions} dimensions, "
            f"{stats.datums} datums"
        )
    except Exception as exc:  # noqa: BLE001 - a JT without PMI is still a valid model
        doc.warn(f"JT PMI: not available ({exc})")


def build_graphic_pmi(path: str):
    """The JT's drawn PMI (frames, leaders, dimension lines, text strokes) as
    the viewer's ``GraphicPmi``.  Groups are named ``jt-<entity id>`` so a
    click maps to the same annotation the semantic overlay created."""
    from qif_viewer.step.graphic_pmi import GraphicPmi

    from .graphic import extract_graphic
    from .pmi import extract_pmi
    from .reader import read_jt

    jt = read_jt(path)
    graphic = extract_graphic(jt)
    names = {e.run_start: f"jt-{e.id}" for e in extract_pmi(jt).entities}
    gp = GraphicPmi()
    for run_start, ent in sorted(graphic.items()):
        lines = [pl.points for pl in ent.polylines if len(pl.points) >= 2]
        if not lines:
            continue
        p0 = len(gp.polylines)
        gp.polylines.extend(lines)
        gp.groups.append((names.get(run_start, f"jt-gfx-{run_start}"), p0, len(gp.polylines), 0, 0))
        gp.occurrence_count += 1
    return gp


def load_jt_file(path: str):
    """Desktop entry: a ``StepLoadResult`` with a pyvista display mesh."""
    import pyvista as pv

    from qif_viewer.step.loader import StepLoadResult

    geom = build_jt_geometry(path)
    pts, tris = geom.points, geom.triangles
    faces_arr = PointsFaces(pts, tris).faces
    mesh = pv.PolyData(pts, faces_arr)
    colors = np.tile(np.asarray(_DEFAULT_MODEL_COLOR, dtype=np.float32), (len(tris), 1))
    mesh.cell_data["FaceColor"] = (colors * 255).clip(0, 255).astype(np.uint8)
    mesh.cell_data["Normals"] = _triangle_normals(pts, tris)
    apply_jt_pmi(geom, mesh, path)

    if geom.edge_polylines:
        counts = [len(c) for c in geom.edge_polylines]
        offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
        lines = np.hstack([np.hstack([[n], np.arange(o, o + n)]) for n, o in zip(counts, offs)]).astype(np.int64)
        edges_mesh = pv.PolyData(np.vstack(geom.edge_polylines), lines=lines)
    else:
        edges_mesh = pv.PolyData()
    try:
        graphic = build_graphic_pmi(path)
    except Exception as exc:  # noqa: BLE001 - drawn PMI is optional
        geom.doc.warn(f"JT graphic PMI: not available ({exc})")
        graphic = None
    result = StepLoadResult(geom.doc, mesh, geom.cell_offsets, edges_mesh, geom.edge_ids, pv.PolyData(), [])
    if graphic is not None:
        result.graphic_pmi = graphic
        geom.doc.warn(f"JT graphic PMI: {graphic.occurrence_count} entities, {len(graphic.polylines)} polylines")
    return result


def _build_faces(scene: JtScene, doc: QifDocument) -> list[tuple[int, int, str]]:
    cell_offsets: list[tuple[int, int, str]] = []
    for i, fg in enumerate(scene.face_groups, start=1):
        face_id, surface_id = f"F{i}", f"S{i}"
        doc.surfaces[surface_id] = OcctSurface(id=surface_id, kind="JtFaceGroup")
        doc.faces[face_id] = Face(
            id=face_id,
            label=fg.name,
            color=_DEFAULT_MODEL_COLOR,
            turned=False,
            surface_id=surface_id,
            loop_ids=[],
        )
        cell_offsets.append((fg.tri_start, fg.tri_end, face_id))
    return cell_offsets


def _triangle_normals(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    np.divide(n, ln, out=n, where=ln > 1e-12)
    return n


def _build_edges(scene: JtScene, doc: QifDocument) -> tuple[list[str], list[np.ndarray]]:
    """Polylines along the borders between face groups (and open borders)."""
    pts, tris = scene.points, scene.triangles
    group_of = np.empty(len(tris), dtype=np.int64)
    for i, fg in enumerate(scene.face_groups):
        group_of[fg.tri_start:fg.tri_end] = i
    normals = _triangle_normals(pts, tris)

    # undirected edge key -> up to two incident triangles
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    tri_of = np.tile(np.arange(len(tris)), 3)
    lo, hi = e.min(axis=1), e.max(axis=1)
    key = lo.astype(np.int64) * (len(pts) + 1) + hi
    order = np.argsort(key, kind="stable")
    key, lo, hi, tri_of = key[order], lo[order], hi[order], tri_of[order]
    new = np.concatenate([[True], key[1:] != key[:-1]])
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], len(key))

    segments: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cos_limit = np.cos(np.radians(_EDGE_MIN_ANGLE_DEG))
    for s, t in zip(starts, ends):
        n = t - s
        a, b = int(lo[s]), int(hi[s])
        if n == 1:
            g = int(group_of[tri_of[s]])
            segments.setdefault((g, -1), []).append((a, b))
        elif n == 2:
            t0, t1 = int(tri_of[s]), int(tri_of[s + 1])
            g0, g1 = int(group_of[t0]), int(group_of[t1])
            if g0 == g1:
                continue
            if float(np.dot(normals[t0], normals[t1])) > cos_limit:
                continue
            segments.setdefault((min(g0, g1), max(g0, g1)), []).append((a, b))

    edge_ids: list[str] = []
    polylines: list[np.ndarray] = []
    for chain in (c for segs in segments.values() for c in _chain(segs)):
        if len(chain) < 2:
            continue
        edge_id = f"E{len(edge_ids) + 1}"
        doc.edges[edge_id] = Edge(id=edge_id, curve_id="", vertex_beg_id="", vertex_end_id="")
        edge_ids.append(edge_id)
        polylines.append(pts[chain])
    return edge_ids, polylines


def _chain(segs: list[tuple[int, int]]) -> list[list[int]]:
    """Join segments sharing end points into polylines."""
    adj: dict[int, list[int]] = {}
    for a, b in segs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used: set[tuple[int, int]] = set()

    def take(a: int, b: int) -> bool:
        k = (a, b) if a < b else (b, a)
        if k in used:
            return False
        used.add(k)
        return True

    def walk(start: int, nxt: int) -> list[int]:
        chain = [start]
        prev, cur = start, nxt
        while True:
            if not take(prev, cur):
                break
            chain.append(cur)
            cand = [x for x in adj[cur] if x != prev and ((x, cur) if x < cur else (cur, x)) not in used]
            if len(adj[cur]) != 2 or not cand:
                break
            prev, cur = cur, cand[0]
        return chain

    out: list[list[int]] = []
    ends = [v for v, n in adj.items() if len(n) != 2]
    for v in ends:
        for w in adj[v]:
            k = (v, w) if v < w else (w, v)
            if k not in used:
                out.append(walk(v, w))
    for v in adj:  # closed loops
        for w in adj[v]:
            k = (v, w) if v < w else (w, v)
            if k not in used:
                out.append(walk(v, w))
    return out
