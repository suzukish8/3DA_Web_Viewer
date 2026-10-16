from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Annotation:
    id: str
    layer: str          # "Datums" or a characteristic type_name (e.g. "Position")
    label: str           # full display text, ready to render
    anchor_point: np.ndarray
    anchor_direction: np.ndarray
    color: tuple[float, float, float]
    measured: Optional[float] = None
    # Structured fields for ui/sidebar.py's ISO 1101 feature control frame
    # rendering -- None for annotations (e.g. Datums) that aren't a ISO GPS
    # characteristic and so have no frame to draw.
    designator: Optional[str] = None
    symbol_kind: Optional[str] = None
    value_text: Optional[str] = None
    datum_refs: list[str] = field(default_factory=list)
    dimension_icon: str = ""
    # Label(s) of Datum(s) defined on this SAME feature (not to be confused
    # with datum_refs, this characteristic's own tolerance's datum
    # references) -- ISO 5459/ASME Y14.5 convention when a feature is both
    # toleranced and itself a datum feature: the datum feature symbol
    # attaches directly below this characteristic's frame instead of
    # getting a second, separate callout. See annotations/builder.py and
    # ui/fcf_widget.py's _AttachedDatumWidget.
    attached_datum_labels: list[str] = field(default_factory=list)
    # 1-based, inclusive -- which lines of the raw QIF XML this entity came
    # from (its defining Datum/CharacteristicNominal element), for the
    # "highlight the source text" UI. None if not found (e.g. XML lacked
    # an id attribute on that element).
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None
    # B-Rep Face id(s) this annotation is actually about (resolved through
    # its referenced FeatureNominal(s)' EntityInternalIds), for
    # click-to-highlight-the-face-on-the-solid. Empty if none resolved.
    face_ids: list[str] = field(default_factory=list)
    # Other characteristics toleranced on this exact same feature (e.g. a
    # Width dimension and a Perpendicularity callout for the same edge) --
    # rendered as additional boxes beside this one, joined by a short
    # connecting line, sharing this Annotation's single leader line/anchor
    # instead of each getting its own separate leader to the same spot.
    # See annotations/builder.py and ui/fcf_widget.py's
    # _CombinedAnnotationsWidget.
    combined_annotations: list["Annotation"] = field(default_factory=list)
    # How many pattern instances this annotation stands for -- either a
    # "dimension" annotation folding in N separate same-value Diameter
    # characteristics, one per hole of an N-hole pattern all governed by
    # one shared Position tolerance, displayed once as "3×⌀3.5±0.1" instead
    # of 3 repeated callouts (see annotations/builder.py's face_ids-based
    # folding); or a symbol-bearing characteristic (Flatness, Position,
    # ...) whose own <FeatureNominalIds n="N"> directly names N same-spec
    # features at once, shown as an "N×" line above its own feature control
    # frame (ui/fcf_widget.FeatureControlFrameWidget, ui/vector_notation.
    # _single_annotation_glyph). 1 for an ordinary, non-pattern annotation
    # (no "N×" prefix shown).
    count: int = 1
    # The two points a linear "size" dimension (Width/Length/Diameter, not
    # Radius) actually spans between -- set only for those types, so
    # ui/viewport.py can render a proper two-arrow dimension line with
    # extension lines instead of a single leader-and-arrow. See
    # annotations/anchors.py's resolve_dimension_endpoints. None for every
    # other annotation kind (geometric tolerances, datums, Radius).
    size_endpoints: Optional[tuple[np.ndarray, np.ndarray]] = None
    # Direction of the outward surface normal at size_endpoints[0] relative
    # to the p1->p2 span: -1 (slab/boss, faces look away from each other),
    # +1 (slot/hole, faces look toward each other), 0 (unknown -- not an
    # opposite-parallel pair). See annotations/anchors.py's resolve_
    # dimension_end_normal_sign; lets ui/viewport.py's "balloon" occlusion
    # pass tell whether the *specific end* a leader is drawn from faces
    # away from the camera.
    size_end_normal_sign: int = 0
    # The QIF-authored nominal/basic size (<TargetValue>) for a Width/
    # Length/Diameter -- e.g. 40.0 for a toleranced "40 +/- 0.2" -- None
    # for every other annotation kind, or a size dimension with no
    # resolvable target (a reference dimension). 2026-09-16, user-directed:
    # lets ui/viewport.py re-derive an idealized Extension Line Start pair
    # exactly nominal_value apart, centred on the true geometric anchor
    # (hole axis / opposite-parallel-planes centre plane -- see
    # annotations/anchors.py's resolve_dimension_endpoints, whose own
    # midpoint(p1, p2) is already exactly that point), instead of the raw
    # measured p1/p2 -- which can be off by whatever the mesh's own
    # tessellation/rounding introduces relative to the officially
    # toleranced value shown in the annotation text.
    size_nominal_value: Optional[float] = None
    # (axis_point, axis_dir [unit], radius) of the CylinderSurface backing
    # a Radius characteristic's own anchor -- None for every other type,
    # or a Radius that doesn't resolve to an actual cylinder. Lets
    # ui/viewport.py recompute a *different* point around the same
    # circumference while the user drags the annotation (see
    # _update_annotation_drag), keeping the leader exactly radial
    # (perpendicular to the surface) at every such point, not just the one
    # originally picked. See annotations/anchors.py's resolve_cylinder_axis.
    cylinder_axis: Optional[tuple[np.ndarray, np.ndarray, float]] = None
    # The fixed geometry (cylinder axis + axial extent, or the two
    # in-plane axes of a parallel-plane pair + their extent) that this
    # annotation's "Balloon & Mirror"-only sub-anchor points are derived
    # from -- set only for the Diameter/Length/Width/Position types, and
    # only when the target resolves to a cylinder or an opposite-parallel-
    # planes pair. The points themselves are computed from this against
    # the annotation's *live* anchor (not baked in here) -- see
    # annotations/anchors.py's resolve_sub_anchor_recipe / sub_anchor_
    # points, and ui/viewport.py's debug_point_table_rows.
    sub_anchor_recipe: Optional[dict] = None
    # A finite set of (point, unit outward normal) positions on this
    # annotation's referenced face(s) that "Balloon" style's automatic
    # anchor re-placement (ui/viewport.py) is allowed to move the leader
    # anchor to -- a bounded candidate set so the per-frame optimizer
    # stays cheap. Camera-independent, resolved once at build time; see
    # annotations/anchors.py's resolve_anchor_candidates. Empty for a
    # feature that resolves to neither a plane nor a cylinder, and for a
    # non-characteristic annotation (datum).
    anchor_candidates: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    # For a size dimension (Diameter/Width/Length) of a hole / slot / rib
    # through a plate: the two faces its dimension line could sit on, as
    # (label, (p1, p2)) -- the centre-of-feature `size_endpoints` slid onto
    # each axial rim. The user picks one via the right-click "Set Dimension
    # Face..." dialog (ui/main_window.py); nothing is auto-applied. []
    # when there is no such choice (thick block, thickness dimension, a
    # non-size annotation). See annotations/anchors.py's
    # dimension_face_options.
    dimension_face_options: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)
