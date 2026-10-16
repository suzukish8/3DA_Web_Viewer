"""Registry dispatching Face+Surface -> triangulated TriangleSoup, keyed on
the concrete Surface subclass.

Adding support for a new analytic/parametric surface type is one
`@register(NewSurfaceType)`-decorated function in a new module, imported at
the bottom of this file. Anything without a registered tessellator (or that
fails during tessellation) degrades to a wireframe-outline fallback instead
of crashing the whole load -- see `unsupported.py`.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np

from qif_viewer.model.brep import Face, Surface

from .base import TriangleSoup

if TYPE_CHECKING:
    pass

SurfaceTessellator = Callable[[Face, Surface, list, list, list], TriangleSoup]

_REGISTRY: dict[type, SurfaceTessellator] = {}


def register(surface_type: type):
    def deco(fn: SurfaceTessellator) -> SurfaceTessellator:
        _REGISTRY[surface_type] = fn
        return fn
    return deco


def tessellate_face(
    face: Face,
    surface: Surface,
    loop_polylines: list[np.ndarray],
    loop_forms: list[str],
    warnings: list[str],
) -> TriangleSoup:
    fn = _REGISTRY.get(type(surface))
    if fn is None:
        from . import unsupported
        warnings.append(
            f"No tessellator registered for surface type {type(surface).__name__} "
            f"(face {face.id}); using wireframe fallback"
        )
        return unsupported.wireframe_fallback(face, loop_polylines)
    try:
        return fn(face, surface, loop_polylines, loop_forms, warnings)
    except Exception as exc:  # noqa: BLE001 - one bad face must not kill the whole load
        from . import unsupported
        warnings.append(f"Face {face.id}: tessellation raised {exc!r}; using wireframe fallback")
        return unsupported.wireframe_fallback(face, loop_polylines)


# Imported for their @register(...) side effects.
from . import planar, cylindrical, conical, nurbs  # noqa: E402,F401
