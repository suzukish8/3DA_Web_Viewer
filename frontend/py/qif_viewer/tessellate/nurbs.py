"""B-spline / NURBS evaluation (Cox-de Boor), shared by the Nurbs13 curve
sampler (curves.py) and the Nurbs23 surface tessellator (surfaces/nurbs.py).
"""
from __future__ import annotations

import numpy as np


def _basis(order: int, knots: np.ndarray, u: float, n_ctrl: int) -> np.ndarray:
    """All n_ctrl B-spline basis values N_{i,order-1}(u) (Cox-de Boor)."""
    p = order - 1
    lo, hi = knots[p], knots[n_ctrl]
    u = min(max(u, lo), hi)
    # Span index: last i with knots[i] <= u < knots[i+1]; clamp to the final
    # non-empty span so u == hi evaluates the end point.
    span = int(np.searchsorted(knots, u, side="right")) - 1
    span = min(max(span, p), n_ctrl - 1)
    N = np.zeros(p + 1)
    N[0] = 1.0
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            den = right[r + 1] + left[j - r]
            tmp = N[r] / den if den != 0.0 else 0.0
            N[r] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        N[j] = saved
    out = np.zeros(n_ctrl)
    out[span - p: span + 1] = N
    return out


def eval_curve(order, knots, ctrl, weights, u: float) -> np.ndarray:
    b = _basis(order, knots, u, len(ctrl))
    if weights is not None:
        b = b * weights
        s = b.sum()
        b = b / s if s != 0.0 else b
    return b @ ctrl


def eval_surface(order_u, order_v, knots_u, knots_v, ctrl, weights, u: float, v: float) -> np.ndarray:
    nu, nv = ctrl.shape[0], ctrl.shape[1]
    bu = _basis(order_u, knots_u, u, nu)
    bv = _basis(order_v, knots_v, v, nv)
    w = np.outer(bu, bv)
    if weights is not None:
        w = w * weights
        s = w.sum()
        w = w / s if s != 0.0 else w
    return np.einsum("ij,ijk->k", w, ctrl)


def sample_params(order: int, knots: np.ndarray, n_ctrl: int, per_span: int) -> np.ndarray:
    """Parameter values: every distinct knot in the valid domain, each span
    subdivided `per_span` times."""
    lo, hi = knots[order - 1], knots[n_ctrl]
    ks = np.unique(np.clip(knots[(knots >= lo) & (knots <= hi)], lo, hi))
    if len(ks) < 2:
        return np.array([lo, hi])
    out = []
    for a, b in zip(ks[:-1], ks[1:]):
        out.append(np.linspace(a, b, per_span, endpoint=False))
    out.append([ks[-1]])
    return np.concatenate(out)
