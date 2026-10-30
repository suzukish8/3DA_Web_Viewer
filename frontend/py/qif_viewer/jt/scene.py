"""Assemble a JT file's tessellated geometry: LSG walk -> Shape LOD decode ->
merged triangle soup in model coordinates, with one face id per B-Rep face
group so the viewer can pick / highlight them like STEP faces."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lsg import read_lsg
from .mesh import MeshError, decode_tristrip_lod
from .reader import JtFile, read_jt


_MARKER_MAX_TRIANGLES = 12
_MARKER_MAX_DIAGONAL = 1.0


@dataclass
class JtFaceGroup:
    shape_index: int
    group: int
    name: str
    tri_start: int
    tri_end: int


@dataclass
class JtScene:
    path: str
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    face_groups: list[JtFaceGroup] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_jt_scene(path: str, jt: JtFile | None = None) -> JtScene:
    jt = jt or read_jt(path)
    scene = JtScene(path)
    lsg = read_lsg(jt)
    seg_by_guid = {s.guid: s for s in jt.segments}

    all_pts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    point_offset = 0
    tri_offset = 0
    for si, ref in enumerate(lsg.shapes):
        seg = seg_by_guid.get(ref.segment_guid)
        if seg is None:
            scene.warnings.append(f"shape {ref.node_id}: segment missing from TOC")
            continue
        body = jt.data[seg.offset + 24:seg.offset + seg.length]
        try:
            m = decode_tristrip_lod(body)
        except (MeshError, ValueError, IndexError) as exc:
            scene.warnings.append(f"shape {ref.node_id} ({ref.name}): {exc}")
            continue
        if not m.triangles:
            continue
        pts = np.asarray(m.vertices, dtype=np.float64)
        homo = np.hstack([pts, np.ones((len(pts), 1))]) @ ref.matrix
        pts = homo[:, :3]
        # NX writes sub-millimetre dummy cubes at the corners of the scene
        # extent; they only distort the model bounds, so skip them.
        if len(m.triangles) <= _MARKER_MAX_TRIANGLES and float(np.linalg.norm(np.ptp(pts, axis=0))) < _MARKER_MAX_DIAGONAL:
            scene.warnings.append(f"shape {ref.node_id}: tiny marker geometry skipped")
            continue
        tris = np.asarray(m.triangles, dtype=np.int64)
        groups = np.asarray(m.groups, dtype=np.int64)
        # A mirrored placement flips the winding.
        if np.linalg.det(ref.matrix[:3, :3]) < 0:
            tris = tris[:, [0, 2, 1]]
        order = np.argsort(groups, kind="stable")
        tris, groups = tris[order], groups[order]
        all_pts.append(pts)
        all_tris.append(tris + point_offset)
        # one contiguous run of triangles per face group
        start = 0
        for k in range(1, len(groups) + 1):
            if k == len(groups) or groups[k] != groups[start]:
                scene.face_groups.append(
                    JtFaceGroup(si, int(groups[start]), ref.name, tri_offset + start, tri_offset + k)
                )
                start = k
        point_offset += len(pts)
        tri_offset += len(tris)

    if all_pts:
        scene.points = np.vstack(all_pts)
        scene.triangles = np.vstack(all_tris)
    return scene
