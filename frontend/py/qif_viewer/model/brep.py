"""B-Rep data model: vertices, curves, edges, loops, faces, surfaces, bodies.

All geometry fields are numpy arrays of shape (3,) in the QIF file's native
linear unit (mm for this sample). Nothing here touches raw XML.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Vertex:
    id: str
    point: np.ndarray


@dataclass
class LineCurve:
    id: str
    start: np.ndarray
    end: np.ndarray


@dataclass
class ArcCurve:
    id: str
    radius: float
    center: np.ndarray
    dir_beg: np.ndarray
    normal: np.ndarray
    domain: tuple[float, float]


@dataclass
class EllipseCurve:
    """ArcConic13 with form="ELLIPSE" (e.g. the elliptical edge where a cone
    meets a non-perpendicular plane, as in a countersink/chamfer)."""
    id: str
    semi_major: float
    semi_minor: float
    center: np.ndarray
    dir_beg: np.ndarray  # major-axis direction (angle=0)
    normal: np.ndarray   # ellipse-plane normal
    domain: tuple[float, float]


@dataclass
class NurbsCurve:
    """Nurbs13: B-spline curve (optionally rational). `ctrl` is (n,3),
    `weights` None or (n,), `knots` has n + order entries."""
    id: str
    order: int
    knots: np.ndarray
    ctrl: np.ndarray
    weights: "np.ndarray | None"
    domain: tuple[float, float]


@dataclass
class UnknownCurve:
    """Placeholder for a curve type/form we can't parametrize yet. Edges
    referencing it fall back to a straight line between their vertices
    (tessellate/edges.py) instead of vanishing."""
    id: str
    xml_tag: str


Curve = Union[LineCurve, ArcCurve, EllipseCurve, NurbsCurve, UnknownCurve]


@dataclass
class CoEdge:
    edge_id: str
    turned: bool


@dataclass
class Loop:
    id: str
    form: str  # "OUTER" | "INNER"
    coedges: list[CoEdge]


@dataclass
class Edge:
    id: str
    curve_id: str
    vertex_beg_id: str
    vertex_end_id: str


@dataclass
class Surface:
    """Base marker for analytic surface types. Never instantiated directly."""
    id: str


@dataclass
class PlaneSurface(Surface):
    origin: np.ndarray
    dir_u: np.ndarray
    dir_v: np.ndarray


@dataclass
class CylinderSurface(Surface):
    diameter: float
    length: float
    axis_point: np.ndarray
    axis_dir: np.ndarray
    dir_beg: np.ndarray
    domain_angle: tuple[float, float]


@dataclass
class ConeSurface(Surface):
    diameter_bottom: float
    diameter_top: float
    length: float
    axis_point: np.ndarray
    axis_dir: np.ndarray
    dir_beg: np.ndarray
    domain_angle: tuple[float, float]


@dataclass
class NurbsSurface(Surface):
    """Nurbs23: tensor-product B-spline surface (optionally rational).
    `ctrl` is (nu, nv, 3), `weights` None or (nu, nv)."""
    order_u: int
    order_v: int
    knots_u: np.ndarray
    knots_v: np.ndarray
    ctrl: np.ndarray
    weights: "np.ndarray | None"


@dataclass
class UnknownSurface(Surface):
    """Placeholder for a surface type we don't have a tessellator for yet.

    Keeps the face in the model (so it still shows up as a wireframe/badge
    fallback) instead of dropping it silently.
    """
    xml_tag: str


@dataclass
class OcctSurface(Surface):
    """Non-plane/cylinder/cone surface from a STEP import (sphere, torus,
    B-spline, surface of revolution/extrusion, etc.), tracked only for
    diagnostics. STEP faces are tessellated directly from OpenCASCADE's own
    triangulation (qif_viewer/step/loader.py), so unlike UnknownSurface this
    never needs a wireframe-fallback tessellator."""
    kind: str


@dataclass
class Face:
    id: str
    label: str
    color: tuple[float, float, float]
    turned: bool
    surface_id: str
    loop_ids: list[str]


@dataclass
class Body:
    id: str
    face_ids: list[str]
    loop_ids: list[str]
    edge_ids: list[str]
    vertex_ids: list[str]
