"""Small geometry helpers shared across tessellate/* and ui/*."""
from __future__ import annotations

import numpy as np

from qif_viewer.model.document import QifDocument


def bbox_diagonal(doc: QifDocument) -> float:
    if not doc.vertices:
        return 1.0
    pts = np.array([v.point for v in doc.vertices.values()])
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def loop_chain_tolerance(doc: QifDocument) -> float:
    """The tolerance used when chaining edges into a loop polyline (and
    when matching duplicated edge copies across faces) -- a small fraction
    of the model's own scale, never smaller than a numerical noise floor."""
    return max(bbox_diagonal(doc) * 1e-4, 1e-6)
