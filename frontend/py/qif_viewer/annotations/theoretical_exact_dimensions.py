"""Theoretical Exact Dimensions (TED -- ISO's term; ASME Y14.5 calls the
same thing a "basic dimension"): the tolerance-free, boxed number giving a
toleranced feature's true location relative to a datum system -- View >
Datum/Datum System > Show Theoretical Exact Dimensions (ui/main_window.py).

Meaningful only relative to a real coordinate system: a DatumReferenceFrame
that isn't geometry/datum_system.py's `fully_constrained` (a lone datum, or
a partial combination that hasn't pinned down all 6 DOF) doesn't fix a real
datum system to measure a "true position" against yet, so a characteristic
whose own CharacteristicDefinition.datum_reference_frame_id resolves to one
of those gets no TED entries at all -- see that module's own docstring for
exactly what "fully_constrained" means.

One TED per (characteristic, datum member) pair -- 2026-09-29, user-
directed: what a single datum actually constrains is what gets dimensioned
against it, nothing more:

- A PLANE datum constrains only the 1 DOF along its own normal -- its own
  TED is the signed distance from that plane to the feature, measured along
  that same normal (the plane's own real geometry, not the DRF's combined
  axes -- see DatumMember).
- An AXIS datum (a cylinder/circle datum feature) constrains only the 2 DOF
  perpendicular to itself, never anything along its own length -- its own
  TED is the perpendicular (radial) distance from that axis line to the
  feature, not a signed offset along any single direction.

A hole against a fully-constrained [A|B|C] datum system (three plane
datums) gets up to three TEDs, one per plane -- dropping any whose value is
negligible (the hole already sits on that particular plane).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qif_viewer.annotations.anchors import resolve_anchor, resolve_cylinder_axis, resolve_sub_anchor_recipe
from qif_viewer.geometry.datum_system import DatumSystemFrame
from qif_viewer.model.document import QifDocument

# A distance below this (model units, i.e. mm) isn't worth its own box --
# floating-point noise from the projection, or a feature that genuinely
# sits on/on-axis-with that particular datum.
_NEGLIGIBLE = 1e-6


@dataclass
class TheoreticalExactDimension:
    characteristic_nominal_id: str
    drf_id: str
    datum_label: str  # which single datum member this is measured from, e.g. "A"
    kind: str  # "PLANE" (a normal-direction distance) or "AXIS" (a perpendicular/radial distance)
    value: float  # PLANE: signed distance along that plane's own normal. AXIS: distance (>=0), always perpendicular.
    foot_point: np.ndarray  # the nearest point ON that plane/axis to the feature -- one end of the dimension line
    feature_point: np.ndarray  # the toleranced feature's own real anchor point -- the other end
    # "DISTANCE" -> a boxed dimension; "PERPENDICULAR" -> a right-angle (⌐)
    # mark where the feature meets the datum plane (the feature's face or
    # axis is square to it, so no distance is toleranced).
    relation: str = "DISTANCE"
    datum_normal: Optional[np.ndarray] = None
    feature_dir: Optional[np.ndarray] = None
    # A hole/cylinder: feature_point is its centre line's point, feature_dir
    # its axis, axis_extent the (start, end) of its axial extent -- what the
    # viewer draws as a chain-line centre line the extension lines meet.
    is_axis: bool = False
    axis_extent: Optional[tuple] = None


def _feature_anchor(
    feature_nominal_ids: list[str], doc: QifDocument,
) -> Optional[tuple[np.ndarray, Optional[np.ndarray], bool, Optional[tuple]]]:
    """(point, direction, is_axis): the toleranced feature's own real
    anchor -- same "first instance, not a pattern centroid" treatment
    annotations/builder.py's own _average_anchor uses. A hole/cylinder is
    located by its CENTER (the rim point resolve_anchor lands on is dropped
    onto the axis) and its direction is the axis (is_axis=True); anything
    else keeps the face anchor and its normal."""
    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        anchor = resolve_anchor(fn, doc)
        if anchor is None:
            continue
        point = np.asarray(anchor[0], dtype=float)
        direction = None if anchor[1] is None else np.asarray(anchor[1], dtype=float)
        cyl = resolve_cylinder_axis([fid], doc)
        if cyl is not None:
            axis_point, axis_dir, _radius = cyl
            t0 = float(np.dot(point - axis_point, axis_dir))
            point = axis_point + axis_dir * t0
            extent = None
            recipe = resolve_sub_anchor_recipe([fid], doc)
            if recipe is not None and recipe.get("kind") == "cylinder":
                d = np.asarray(recipe["axis_dir"], dtype=float)
                a0 = np.asarray(recipe["axis_point"], dtype=float)
                t_here = float(np.dot(point - a0, d))
                extent = (
                    point + d * (float(recipe["proj_min"]) - t_here),
                    point + d * (float(recipe["proj_max"]) - t_here),
                )
            return point, axis_dir, True, extent
        return point, direction, False, None
    return None


def compute_theoretical_exact_dimensions(
    doc: QifDocument, datum_systems: dict[str, DatumSystemFrame],
) -> list[TheoreticalExactDimension]:
    """One entry per (characteristic, datum member). 2026-10-15,
    user-directed: any datum reference frame with resolved members counts
    (not just a fully constrained one), and a plane datum yields either a
    distance (feature parallel to it) or a right-angle mark (feature square
    to it); oblique features give nothing."""
    results: list[TheoreticalExactDimension] = []
    for nominal in doc.characteristic_nominals.values():
        definition = doc.characteristic_definitions.get(nominal.definition_id)
        if definition is None or not definition.datum_reference_frame_id:
            continue
        frame = datum_systems.get(definition.datum_reference_frame_id)
        if frame is None or not frame.members:
            continue
        found = _feature_anchor(nominal.feature_nominal_ids, doc)
        if found is None:
            continue
        point, fdir, is_axis, axis_extent = found

        for member in frame.members:
            if member.kind == "PLANE":
                nd = np.asarray(member.direction, dtype=float)
                nd = nd / (np.linalg.norm(nd) or 1.0)
                signed = float(np.dot(point - member.point, nd))
                foot = point - nd * signed
                relation = "DISTANCE"
                if fdir is not None and np.linalg.norm(fdir) > 1e-9:
                    c = abs(float(np.dot(fdir / np.linalg.norm(fdir), nd)))
                    if is_axis:
                        # axis along the normal -> square to the plane; axis
                        # inside the plane's direction -> a distance.
                        relation = "PERPENDICULAR" if c > 0.999 else ("DISTANCE" if c < 0.01 else "")
                    else:
                        # face normal along the datum normal -> parallel
                        # faces (distance); normals square -> faces square.
                        relation = "DISTANCE" if c > 0.999 else ("PERPENDICULAR" if c < 0.01 else "")
                if not relation:
                    continue
                if relation == "DISTANCE" and abs(signed) < _NEGLIGIBLE:
                    continue
                results.append(TheoreticalExactDimension(
                    characteristic_nominal_id=nominal.id, drf_id=frame.id, datum_label=member.label,
                    kind="PLANE", value=signed, foot_point=foot, feature_point=point,
                    relation=relation, datum_normal=nd,
                    feature_dir=None if fdir is None else np.asarray(fdir, dtype=float),
                    is_axis=is_axis, axis_extent=axis_extent,
                ))
            else:  # "AXIS" -- perpendicular (radial) distance, never along the axis itself
                along = float(np.dot(point - member.point, member.direction))
                closest_on_axis = member.point + member.direction * along
                value = float(np.linalg.norm(point - closest_on_axis))
                if value < _NEGLIGIBLE:
                    continue
                results.append(TheoreticalExactDimension(
                    characteristic_nominal_id=nominal.id, drf_id=frame.id, datum_label=member.label,
                    kind="AXIS", value=value, foot_point=closest_on_axis, feature_point=point,
                ))
    return results
