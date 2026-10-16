"""Edge -> 3D polyline, with a straight-line vertex fallback for any curve
type `curves.sample_curve` doesn't recognize."""
from __future__ import annotations

import numpy as np

from qif_viewer.model.brep import Edge
from qif_viewer.model.document import QifDocument

from .curves import sample_curve


def polyline_for_edge(edge: Edge, doc: QifDocument, warnings: list[str]) -> np.ndarray:
    curve = doc.curves.get(edge.curve_id)
    pts = sample_curve(curve) if curve is not None else None
    if pts is not None:
        return pts

    v_beg = doc.vertices.get(edge.vertex_beg_id)
    v_end = doc.vertices.get(edge.vertex_end_id)
    if v_beg is None or v_end is None:
        warnings.append(f"Edge {edge.id}: no curve geometry and no resolvable vertices; dropped")
        return np.zeros((0, 3))
    warnings.append(f"Edge {edge.id}: curve type unsupported, using straight vertex-to-vertex fallback")
    return np.array([v_beg.point, v_end.point])
