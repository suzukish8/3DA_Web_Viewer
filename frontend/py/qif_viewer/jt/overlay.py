"""Overlay a JT file's semantic PMI onto B-rep display geometry.

The geometry is either the JT's own tessellation (``jt/loader.py``) or, when
a same-named STEP sits next to it, that STEP's exact B-rep.  JT leader
terminator points share the geometry's model coordinates; each is snapped to
the nearest face, which becomes the annotation's anchor face.

The result is written into the same ``QifDocument`` fields
``qif_viewer.step.pmi.populate_pmi`` fills, so the existing annotation
builder/viewport need no JT-specific code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from qif_viewer.model.document import QifDocument
from qif_viewer.model.features import GenericFeatureNominal
from qif_viewer.model.gdt import (
    CharacteristicDefinition,
    CharacteristicNominal,
    Datum,
    DatumReferenceFrame,
)

from .pmi import extract_pmi
from .reader import read_jt
from .semantic import JtCharacteristic, JtDatum, JtDimension, to_semantic

_STEP_SUFFIXES = (".stp", ".step", ".STP", ".STEP")


@dataclass
class OverlayStats:
    characteristics: int = 0
    dimensions: int = 0
    datums: int = 0
    off_surface: int = 0          # anchors that had to fall back to text origin
    max_snap_distance: float = 0.0


def find_companion_step(jt_path: str) -> Optional[str]:
    """Same-stem STEP next to the JT, else in a sibling directory (the JAMA
    sample keeps ``03knuckle`` and ``20_Knuckle_STEP`` side by side)."""
    jt = Path(jt_path)
    stem = jt.stem.lower()
    for folder in (jt.parent, *[d for d in jt.parent.parent.iterdir() if d.is_dir()]):
        try:
            for f in folder.iterdir():
                if f.is_file() and f.stem.lower() == stem and f.suffix.lower() in (".stp", ".step"):
                    return str(f)
        except OSError:
            continue
    return None


def _closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Vectorised closest point of ``p`` on every triangle (Ericson, RTCD 5.1.5).
    Returns (points (n,3), squared distances (n,))."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        v_in, w_in = vb * denom, vc * denom
        v_ab, v_ac = d1 / (d1 - d3), d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    q = a + ab * v_in[:, None] + ac * w_in[:, None]          # interior
    q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], a + ab * v_ab[:, None], q)
    q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], a + ac * v_ac[:, None], q)
    q = np.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[:, None], b + (c - b) * w_bc[:, None], q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, q)       # vertex regions
    q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)
    diff = q - p
    return q, np.einsum("ij,ij->i", diff, diff)


class _Surface:
    """Closest-point queries on the display mesh, with face ids.  Uses a VTK
    cell locator when VTK is installed, else a vectorised numpy search (the
    browser build runs without VTK)."""

    def __init__(self, mesh, cell_offsets):
        try:
            import vtk
        except ImportError:
            vtk = None
        if not hasattr(mesh, "GetNumberOfCells"):  # a plain (points, faces) holder, not a VTK dataset
            vtk = None
        self._vtk = vtk
        self._mesh = mesh
        self._points = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces)
        self._tris = faces.reshape(-1, 4)[:, 1:4] if faces.size and faces.size % 4 == 0 else None
        if vtk is not None:
            self._loc = vtk.vtkCellLocator()
            self._loc.SetDataSet(mesh)
            self._loc.BuildLocator()
        self._ranges = cell_offsets  # (start, end, face_id)
        self._starts = np.array([r[0] for r in cell_offsets], dtype=np.int64)

    def _nearest_cell(self, p) -> tuple[np.ndarray, int, float]:
        if self._vtk is not None:
            vtk = self._vtk
            cp = [0.0, 0.0, 0.0]
            cid, sub, d2 = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
            self._loc.FindClosestPoint([float(p[0]), float(p[1]), float(p[2])], cp, cid, sub, d2)
            return np.array(cp), int(cid.get()), float(d2.get())
        pts, t = self._points, self._tris
        q, d2 = _closest_on_triangles(np.asarray(p, dtype=float), pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]])
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        k = int(np.argmin(d2))
        return q[k], k, float(d2[k])

    def closest(self, p) -> tuple[np.ndarray, np.ndarray, Optional[str], float]:
        cp, cell, d2 = self._nearest_cell(p)
        normal = np.array([0.0, 0.0, 1.0])
        if self._tris is not None and 0 <= cell < len(self._tris):
            a, b, c = self._points[self._tris[cell]]
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                normal = n / ln
        face_id = None
        if len(self._starts):
            k = int(np.searchsorted(self._starts, cell, side="right")) - 1
            if 0 <= k < len(self._ranges) and self._ranges[k][0] <= cell < self._ranges[k][1]:
                face_id = self._ranges[k][2]
        return np.array(cp), normal, face_id, math.sqrt(max(d2, 0.0))


def _fmt(v: float) -> str:
    return f"{v:g}"


def overlay_jt_pmi(doc: QifDocument, mesh, cell_offsets, jt_path: str) -> OverlayStats:
    """Replace ``doc``'s PMI (whatever the STEP pass produced) with the JT's."""
    stats = OverlayStats()
    try:  # shared placeholder element (the STEP module needs OCP, absent in the browser build)
        from qif_viewer.step.pmi import _EMPTY_ELEMENT
    except ImportError:
        import xml.etree.ElementTree as ET

        _EMPTY_ELEMENT = ET.Element("StepPMI")

    sem = to_semantic(extract_pmi(read_jt(jt_path)))
    surf = _Surface(mesh, cell_offsets)

    doc.characteristic_definitions.clear()
    doc.characteristic_nominals.clear()
    doc.characteristic_items.clear()
    doc.datums.clear()
    doc.datum_reference_frames.clear()
    doc.feature_nominals.clear()
    name_index: dict[str, list[str]] = {}
    doc.step_pmi_name_index = name_index

    # Pass 1: exact anchors.  A leader that carries an NX ``Reference`` ends
    # on the toleranced geometry, so its terminator snaps to the STEP
    # surface at distance ~0.  An unreferenced terminator only counts when
    # it happens to lie on the surface already.
    _ON_SURFACE_TOL = 0.5
    everything = [*sem.characteristics, *sem.dimensions, *sem.datums]
    exact: dict[int, tuple] = {}
    for e in everything:
        for p, ref in e.leaders:
            if p is None:
                continue
            q, n, face_id, dist = surf.closest(p)
            if (ref and dist < 5.0) or dist < _ON_SURFACE_TOL:
                exact[e.jt_id] = (q, n, face_id)
                stats.max_snap_distance = max(stats.max_snap_distance, dist)
                break

    # Pass 2: frames drawn off a dimension line / another symbol have only
    # free-space terminators.  They are attached to whichever anchored
    # entity lies closest to their own stub, which is what the CAD system's
    # drawing placed them against.
    ref_points: list[tuple[np.ndarray, int]] = []
    for e in everything:
        if e.jt_id in exact:
            pts = [p for p, _ in e.leaders if p is not None]
            if e.text_origin is not None:
                pts.append(e.text_origin)
            ref_points.extend((np.asarray(p, dtype=float), e.jt_id) for p in pts)
    ref_arr = np.array([p for p, _ in ref_points]) if ref_points else np.empty((0, 3))

    def anchor(entity):
        """-> (location, normal, face_id) on the STEP surface, or None."""
        if entity.jt_id in exact:
            return exact[entity.jt_id]
        probes = [p for p, _ in entity.leaders if p is not None]
        if entity.text_origin is not None:
            probes.append(entity.text_origin)
        stats.off_surface += 1
        if len(ref_arr) and probes:
            best = min(
                ((np.linalg.norm(ref_arr - np.asarray(q, dtype=float), axis=1), q) for q in probes),
                key=lambda t: t[0].min(),
            )
            return exact[ref_points[int(best[0].argmin())][1]]
        if probes:
            q, n, face_id, dist = surf.closest(probes[0])
            stats.max_snap_distance = max(stats.max_snap_distance, dist)
            return q, n, face_id
        return None

    def add_fn(prefix: str, jt_id: int, a) -> str:
        fn_id = f"{prefix}{jt_id}"
        doc.feature_nominals[fn_id] = GenericFeatureNominal(
            id=fn_id, name="", kind="JtPMI", location=a[0], axis_dir=a[1],
            raw_element=_EMPTY_ELEMENT,
            entity_internal_ids=[a[2]] if a[2] else [],
        )
        return fn_id

    # --- datums: feature symbols first, then targets for labels without one.
    letter_to_id: dict[str, str] = {}

    def datum_id(label: str) -> str:
        did = letter_to_id.get(label)
        if did is None:
            did = f"DATUM_{label}"
            doc.datums[did] = Datum(id=did, label=label, feature_nominal_ids=[])
            letter_to_id[label] = did
        return did

    ordered = sorted(sem.datums, key=lambda d: d.is_target)
    for d in ordered:
        if not d.label:
            continue
        did = datum_id(d.label)
        if doc.datums[did].feature_nominal_ids:
            continue
        a = anchor(d)
        if a is None:
            continue
        doc.datums[did].feature_nominal_ids = [add_fn("FN_JDF", d.jt_id, a)]
        stats.datums += 1

    # --- geometric tolerances
    seq = 0
    for c in sem.characteristics:
        a = anchor(c)
        if a is None:
            doc.warn(f"JT PMI: {c.type_name} (id {c.jt_id}) has no usable anchor; skipped")
            continue
        seq += 1
        levels = [("PRIMARY", c.primary), ("SECONDARY", c.secondary), ("TERTIARY", c.tertiary)]
        entries = [(datum_id(r.label), prec) for prec, refs in levels for r in refs]
        drf_id = None
        if entries:
            drf_id = f"JDRF_{c.jt_id}"
            doc.datum_reference_frames[drf_id] = DatumReferenceFrame(id=drf_id, entries=entries)
        def_id = f"JDEF_{c.jt_id}"
        doc.characteristic_definitions[def_id] = CharacteristicDefinition(
            id=def_id, type_name=c.type_name, tolerance_value=c.tolerance_value,
            datum_reference_frame_id=drf_id, material_condition=c.material_condition,
            zone_shape=c.zone_shape, common_zone=c.common_zone,
        )
        nom_id = f"JNOM_{c.jt_id}"
        fn_id = add_fn("FN_JGT", c.jt_id, a)
        doc.characteristic_nominals[nom_id] = CharacteristicNominal(
            id=nom_id, type_name=c.type_name, definition_id=def_id,
            feature_nominal_ids=[fn_id], designator=str(seq),
        )
        name_index.setdefault(f"jt-{c.jt_id}", []).append(f"char-{nom_id}")
        stats.characteristics += 1

    # --- dimensions
    for dm in sem.dimensions:
        value = dm.value
        m = re.match(r"^\s*([\d.]+)", dm.text or "")
        nominal = float(m.group(1)) if m else value
        if nominal is None:
            continue
        angular = dm.type_code == 5 or "<$s>" in (dm.text or "")
        pts = [(p, r) for p, r in dm.leaders if p is not None and r]
        if len(pts) >= 2:
            qa = surf.closest(pts[0][0])
            qb = surf.closest(pts[-1][0])
            a = ((qa[0] + qb[0]) / 2.0, qa[1], qa[2])
        else:
            a = anchor(dm)
        if a is None:
            continue
        seq += 1
        type_name = "Length" if (len(pts) >= 2 or angular) else "Width"
        raw_extra: dict[str, str] = {}
        if dm.upper_delta is not None and dm.lower_delta is not None:
            raw_extra["Tolerance.MaxValue"] = _fmt(dm.upper_delta)
            raw_extra["Tolerance.MinValue"] = _fmt(dm.lower_delta)
        else:
            raw_extra["NonTolerance"] = "MEASURED"
        shown = nominal if not angular else (nominal if m else math.degrees(value or 0.0))
        target = f"∠{_fmt(round(shown, 4))}°" if angular else _fmt(round(shown, 4))
        def_id = f"JDEF_{dm.jt_id}"
        doc.characteristic_definitions[def_id] = CharacteristicDefinition(
            id=def_id, type_name=type_name, tolerance_value=None,
            datum_reference_frame_id=None, material_condition=None, zone_shape=None,
            raw_extra=raw_extra,
        )
        nom_id = f"JNOM_{dm.jt_id}"
        fn_id = add_fn("FN_JDIM", dm.jt_id, a)
        doc.characteristic_nominals[nom_id] = CharacteristicNominal(
            id=nom_id, type_name=type_name, definition_id=def_id,
            feature_nominal_ids=[fn_id], designator=str(seq),
            raw_extra={"TargetValue": target},
        )
        name_index.setdefault(f"jt-{dm.jt_id}", []).append(f"char-{nom_id}")
        stats.dimensions += 1

    doc.warn(
        f"JT PMI ({Path(jt_path).name}): {stats.characteristics} geometric tolerances, "
        f"{stats.dimensions} dimensions, {stats.datums} datums; "
        f"{stats.off_surface} unreferenced frames attached to the nearest anchored PMI; "
        f"max snap distance {stats.max_snap_distance:.3g}"
    )
    return stats
