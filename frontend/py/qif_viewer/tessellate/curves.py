"""Samples QIF curve entities into 3D polylines.

A registry dispatches on curve type so a future curve type (e.g. a general
BSplineCurve) just needs a new `@register(...)` function here. `sample_curve`
returns None for anything unregistered -- the caller (tessellate/edges.py)
falls back to a straight line between the edge's own vertices, so one odd
curve degrades one edge, never the whole model.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from qif_viewer.model.brep import ArcCurve, Curve, EllipseCurve, LineCurve, NurbsCurve

from .nurbs import eval_curve, sample_params

_SAMPLERS: dict[type, Callable[[Curve], np.ndarray]] = {}


def register(curve_type: type):
    def deco(fn: Callable[[Curve], np.ndarray]) -> Callable[[Curve], np.ndarray]:
        _SAMPLERS[curve_type] = fn
        return fn
    return deco


def sample_curve(curve: Curve) -> Optional[np.ndarray]:
    fn = _SAMPLERS.get(type(curve))
    return fn(curve) if fn is not None else None


def segments_for_sweep(sweep_radians: float, per_full_turn: int = 64, minimum: int = 6) -> int:
    turns = abs(sweep_radians) / (2 * math.pi)
    if turns < 1e-9:
        return minimum
    return max(minimum, round(per_full_turn * turns))


def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1 - math.cos(angle))
    )


@register(LineCurve)
def _sample_line(curve: LineCurve) -> np.ndarray:
    return np.array([curve.start, curve.end])


@register(ArcCurve)
def _sample_arc(curve: ArcCurve) -> np.ndarray:
    # `DirBeg` is the radial direction at the underlying full circle's
    # angle-zero reference, and `domain` gives the trim range measured in
    # that same absolute angular coordinate -- NOT an angle relative to the
    # domain's own start. Rotating by the raw domain value (not `th - t0`)
    # is what actually lands on the edge's real start/end vertices; verified
    # against real QIF data where t0 != 0.
    t0, t1 = curve.domain
    n = segments_for_sweep(t1 - t0)
    thetas = np.linspace(t0, t1, n + 1)
    return np.array([curve.center + curve.radius * rodrigues_rotate(curve.dir_beg, curve.normal, th) for th in thetas])


@register(EllipseCurve)
def _sample_ellipse(curve: EllipseCurve) -> np.ndarray:
    # Same absolute-angle convention as _sample_arc above: `dir_beg` is the
    # ellipse's major-axis direction (parameter angle 0), and `domain`
    # values are absolute angles in that frame.
    u = curve.dir_beg / np.linalg.norm(curve.dir_beg)
    n = curve.normal / np.linalg.norm(curve.normal)
    v = np.cross(n, u)
    t0, t1 = curve.domain
    n_segs = segments_for_sweep(t1 - t0)
    thetas = np.linspace(t0, t1, n_segs + 1)
    return np.array([
        curve.center + curve.semi_major * math.cos(th) * u + curve.semi_minor * math.sin(th) * v
        for th in thetas
    ])


@register(NurbsCurve)
def _sample_nurbs(curve: NurbsCurve) -> np.ndarray:
    n = len(curve.ctrl)
    # Order-2 is a polyline (knots suffice); otherwise aim for ~48 samples
    # in total -- densely knotted curves are already smooth at 1 per span.
    spans = max(1, n - curve.order + 1)
    per_span = 1 if curve.order <= 2 else max(1, -(-48 // spans))
    params = sample_params(curve.order, curve.knots, n, per_span)
    return np.array([eval_curve(curve.order, curve.knots, curve.ctrl, curve.weights, u) for u in params])
