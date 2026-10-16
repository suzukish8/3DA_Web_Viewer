"""Backend-agnostic polygon-with-holes triangulation.

Prefers `mapbox_earcut` (small prebuilt wheel, fast, battle-tested). Falls
back automatically to the vendored pure-Python implementation if the wheel
isn't importable on a given machine, so the app never hard-depends on a
compiled package succeeding.
"""
from __future__ import annotations

import logging

import numpy as np

_log = logging.getLogger(__name__)

try:
    import mapbox_earcut as _me

    def triangulate(verts2d: np.ndarray, ring_ends: np.ndarray) -> np.ndarray:
        return _me.triangulate_float64(verts2d.astype(np.float64), ring_ends.astype(np.uint32))

except ImportError:  # pragma: no cover - exercised only when the wheel is unavailable
    _log.warning("mapbox_earcut not importable; using slower pure-Python earcut fallback")
    from ._earcut_py import triangulate as _fallback_triangulate

    def triangulate(verts2d: np.ndarray, ring_ends: np.ndarray) -> np.ndarray:
        return _fallback_triangulate(verts2d, ring_ends)
