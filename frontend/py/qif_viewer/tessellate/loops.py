"""Chains a Loop's CoEdges into a single closed 3D polyline.

Edges are matched by shared 3D endpoints -- the primary, verifiable signal.
`CoEdge.turned` is only consulted when endpoint matching is ambiguous (the
gap exceeds tolerance in both candidate orientations), since its exact
semantics aren't 100% certain from spec-reading alone; disagreements are
logged so a systematic mismatch would show up in the warnings log.
"""
from __future__ import annotations

import numpy as np

from qif_viewer.model.brep import Loop
from qif_viewer.model.document import QifDocument

from .edges import polyline_for_edge


def _chain(segments: list[tuple[np.ndarray, bool]], seed_reversed: bool, tol: float):
    """Chains `segments` (in their given, QIF-specified traversal order)
    starting from `segments[0]` in either its natural or reversed direction.

    Returns (ordered_polylines, total_gap_error, local_warnings). The caller
    tries both seed orientations and keeps whichever produces the smaller
    total error, since the *first* edge has no predecessor to disambiguate
    its direction from -- a wrong guess there would otherwise cascade
    "ambiguous" mismatches down the entire rest of the loop.
    """
    first_pts, _ = segments[0]
    ordered: list[np.ndarray] = [first_pts[::-1] if seed_reversed else first_pts]
    total_error = 0.0
    local_warnings: list[str] = []

    for pts, turned_hint in segments[1:]:
        tail = ordered[-1][-1]
        fwd_gap = float(np.linalg.norm(pts[0] - tail))
        rev_gap = float(np.linalg.norm(pts[-1] - tail))
        best_gap = min(fwd_gap, rev_gap)
        if best_gap <= tol:
            chosen = pts if fwd_gap <= rev_gap else pts[::-1]
        else:
            # Endpoint matching is ambiguous beyond tolerance -- fall back to
            # the `turned` hint and note the disagreement.
            chosen = pts[::-1] if turned_hint else pts
            local_warnings.append(
                f"edge endpoint gap {best_gap:.4g} exceeds tolerance {tol:.4g}; used CoEdge.turned hint instead"
            )
        total_error += best_gap
        ordered.append(chosen)

    return ordered, total_error, local_warnings


def loop_to_polyline(loop: Loop, doc: QifDocument, warnings: list[str], tol: float) -> np.ndarray:
    segments: list[tuple[np.ndarray, bool]] = []
    for coedge in loop.coedges:
        edge = doc.edges.get(coedge.edge_id)
        if edge is None:
            warnings.append(f"Loop {loop.id}: missing edge {coedge.edge_id}; skipped")
            continue
        pts = polyline_for_edge(edge, doc, warnings)
        if len(pts) < 2:
            continue
        segments.append((pts, coedge.turned))

    if not segments:
        return np.zeros((0, 3))
    if len(segments) == 1:
        ordered = [segments[0][0]]
    else:
        as_is = _chain(segments, seed_reversed=False, tol=tol)
        reversed_seed = _chain(segments, seed_reversed=True, tol=tol)
        ordered, _, local_warnings = as_is if as_is[1] <= reversed_seed[1] else reversed_seed
        for msg in local_warnings:
            warnings.append(f"Loop {loop.id}: {msg}")

    polyline = np.concatenate([ordered[0]] + [seg[1:] for seg in ordered[1:]], axis=0)
    if len(polyline) > 1 and np.linalg.norm(polyline[0] - polyline[-1]) < tol:
        polyline = polyline[:-1]
    return polyline
