"""QifDocument -> list[Annotation]: datum labels plus every ISO GPS
characteristic instance, each resolved to a 3D anchor and a display label."""
from __future__ import annotations

from typing import Optional

import numpy as np

from qif_viewer.model.document import QifDocument
from qif_viewer.model.gdt import CharacteristicDefinition, CharacteristicNominal

from . import types_gdt  # noqa: F401 - side-effect: registers Position/Flatness/SurfaceProfile/Width
from .anchors import (
    count_feature_instances,
    dimension_face_options,
    resolve_anchor,
    resolve_anchor_candidates,
    resolve_cylinder_axis,
    resolve_dimension_end_normal_sign,
    resolve_dimension_endpoints,
    resolve_face_ids,
    resolve_sub_anchor_recipe,
)
from .model import Annotation
from .registry import get_spec
from .types_gdt import tolerance_value_text

_PRECEDENCE_ORDER = {"PRIMARY": 0, "SECONDARY": 1, "TERTIARY": 2}
_DATUM_COLOR = (0.1, 0.1, 0.85)

# Radius and Length are excluded from the same-feature combining below --
# unlike a Width/Diameter dimension (commonly paired with a position/
# perpendicularity callout on the very same feature), a Radius or Length
# value stays its own separate, standalone callout even when a tolerance
# frame happens to apply to the same feature (e.g. a fillet's basic radius
# next to its SurfaceProfile tolerance).
_NEVER_COMBINE_TYPES = {"Radius", "Length"}

# ISO 129/ASME Y14.5: a true linear "size" dimension (Width/Diameter/
# Length -- how big a feature is) is drawn as a two-arrow dimension line
# with extension lines to the two points it spans between, not a single
# leader-and-arrow pointing at just one location. Radius is excluded --
# it's measured from an implied center to one edge, so ISO 1101 draws it
# with a single arrow, same as the geometric-tolerance leaders.
_SIZE_DIMENSION_TYPES = {"Width", "Diameter", "Length"}

# Characteristic types that get "Balloon & Mirror"-only sub-anchor points
# (Annotation.sub_anchor_recipe, View > Debug > Point Overlays > Show Sub
# Anchor Points) -- see annotations/anchors.py's resolve_sub_anchor_recipe.
_SUB_ANCHOR_TYPES = {"Diameter", "Length", "Width", "Position"}

# ISO 1101 material-condition modifiers, shown as a trailing letter in the
# tolerance compartment. The real symbol is a circled M/L, but a circled-
# letter glyph (e.g. U+24C2) isn't reliably rendered by the default font on
# an arbitrary Windows install (see fcf_widget.py's module docstring for
# why this project hand-draws its ISO GPS symbols instead of using Unicode
# ones) -- "(M)"/"(L)" is the standard fallback notation when a circled
# glyph isn't available. RFS ("regardless of feature size") and an absent/
# "NONE" condition both get no modifier at all -- it's the default.
_MATERIAL_MODIFIER = {"MAXIMUM": "Ⓜ", "MMC": "Ⓜ", "MINIMUM": "Ⓛ", "LMC": "Ⓛ"}


def _datum_refs(doc: QifDocument, drf_id: Optional[str]) -> list[str]:
    if drf_id is None:
        return []
    drf = doc.datum_reference_frames.get(drf_id)
    if drf is None or not drf.entries:
        return []
    entries = sorted(drf.entries, key=lambda e: _PRECEDENCE_ORDER.get(e[1], 99))
    return [doc.datums[d_id].label for d_id, _ in entries if d_id in doc.datums]


def _iso_value_text(definition: CharacteristicDefinition, nominal: CharacteristicNominal) -> str:
    """The feature control frame's tolerance-compartment text: the core
    value plus its ISO 1101 modifiers (diameter zone, material condition,
    common zone) -- everything except the datum references, which get
    their own compartments.

    For a size characteristic like Width, "the core value" on its own is
    just the +/- tolerance -- the nominal (basic) size it's measured from,
    e.g. the "40" in "40 +0.2/-0.2", is a separate <TargetValue> on the
    nominal rather than anything on the definition, and is prepended here
    when present."""
    # "MEASURED" (QIF's <NonTolerance> value for a reference dimension with
    # no tolerance at all) isn't itself a value worth displaying -- the
    # actual number the user cares about is the nominal/target value next
    # to it (e.g. "1" in a Radius's "R1"), so it's dropped rather than
    # shown as literal text.
    text = tolerance_value_text(definition)
    if text == "MEASURED":
        text = ""
    target = nominal.raw_extra.get("TargetValue")
    if target is not None:
        try:
            target = f"{float(target):g}"
        except ValueError:
            pass  # non-numeric -- show whatever the file actually says
        text = f"{target} {text}".strip()
    if definition.zone_shape == "DIAMETRICAL":
        text = f"⌀{text}"
    modifier = _MATERIAL_MODIFIER.get((definition.material_condition or "").upper())
    if modifier:
        text += modifier
    if definition.common_zone:
        text += " CZ"
    return text


def _average_anchor(feature_nominal_ids: list[str], doc: QifDocument) -> Optional[tuple[np.ndarray, np.ndarray]]:
    anchors = []
    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        anchor = resolve_anchor(fn, doc)
        if anchor is not None:
            anchors.append(anchor)
    if not anchors:
        return None
    if len(anchors) > 1:
        # A characteristic applying uniformly to a whole pattern of
        # distinct features (e.g. one Diameter tolerance called out once
        # for 4 separate holes) references all of them here -- but
        # averaging their anchor points together lands at the pattern's
        # geometric center, off in space between the actual holes, not on
        # any of them. Anchor to the first one instead, the same
        # "point at one real instance, not a centroid of several" fix
        # applied in annotations/anchors.py for a single feature spanning
        # multiple faces.
        return anchors[0]
    point = np.mean([a[0] for a in anchors], axis=0)
    direction = np.mean([a[1] for a in anchors], axis=0)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    return point, direction


def _union_face_ids(feature_nominal_ids: list[str], doc: QifDocument) -> list[str]:
    face_ids: list[str] = []
    seen: set[str] = set()
    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        for face_id in resolve_face_ids(fn, doc):
            if face_id not in seen:
                seen.add(face_id)
                face_ids.append(face_id)
    return face_ids


def _total_feature_instances(feature_nominal_ids: list[str], doc: QifDocument) -> int:
    """Sum of count_feature_instances over every top-level FeatureNominalIds
    entry -- see that function's own docstring for why this, not
    len(_union_face_ids(...)), is the right basis for a "dimension"-kind
    (Diameter/Radius) characteristic's "N×" pattern count."""
    total = 0
    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        total += count_feature_instances(fn, doc)
    return total


def _has_symbol(symbol_kind: Optional[str]) -> bool:
    """True for a characteristic type ISO 1101 draws with an actual boxed
    symbol (Position/Flatness/...), false for "dimension" (Width/Diameter/
    Radius/Length -- plain "<icon> value" text, no box) and for None
    (non-characteristic annotations). Mirrors ui/fcf_widget.has_symbol();
    kept as its own copy here rather than imported so annotations/ doesn't
    depend on ui/ -- both must stay in sync if a new non-boxed
    symbol_kind is ever registered."""
    return symbol_kind is not None and symbol_kind != "dimension"


def build_annotations(doc: QifDocument) -> list[Annotation]:
    annotations: list[Annotation] = []

    items_by_nominal: dict[str, list] = {}
    for item in doc.characteristic_items.values():
        items_by_nominal.setdefault(item.nominal_id, []).append(item)

    # Characteristics that all tolerance the exact same feature(s) --
    # e.g. a Width dimension and a Perpendicularity callout for the same
    # edge -- are combined into one annotation (Annotation.combined_
    # annotations) with a single shared leader line, rather than several
    # separate leaders all converging on the same point on the part.
    # Built per-nominal (not up front) since it only makes sense for
    # nominals that actually end up with a resolvable annotation below.
    annotation_by_nominal_id: dict[str, Annotation] = {}

    for nominal in doc.characteristic_nominals.values():
        definition = doc.characteristic_definitions.get(nominal.definition_id)
        if definition is None:
            doc.warn(f"CharacteristicNominal {nominal.id}: missing definition {nominal.definition_id}; skipped")
            continue

        anchor = _average_anchor(nominal.feature_nominal_ids, doc)
        if anchor is None:
            doc.warn(f"CharacteristicNominal {nominal.id} ({nominal.type_name}): no resolvable anchor; skipped")
            continue

        item = next(iter(items_by_nominal.get(nominal.id, [])), None)
        spec = get_spec(nominal.type_name, doc.warnings)
        value_text = spec.label_formatter(nominal, definition, item)
        datum_refs = _datum_refs(doc, definition.datum_reference_frame_id)
        label = f"{spec.symbol_ascii} {nominal.designator}: {value_text}"
        if datum_refs:
            label += f" [{'|'.join(datum_refs)}]"

        size_endpoints = None
        size_end_normal_sign = 0
        nominal_value = None
        if nominal.type_name in _SIZE_DIMENSION_TYPES:
            target = nominal.raw_extra.get("TargetValue")
            try:
                nominal_value = float(target) if target is not None else None
            except ValueError:
                nominal_value = None
            size_endpoints = resolve_dimension_endpoints(
                nominal.feature_nominal_ids, doc, nominal_value, nominal.type_name
            )
            size_end_normal_sign = resolve_dimension_end_normal_sign(
                nominal.feature_nominal_ids, doc, nominal_value, nominal.type_name
            )

        # ユーザー指示: Radius の引出線は対象の円筒面に常に垂直（＝半径
        # 方向）でなければならない -- 角度方向にドラッグで動かした後も
        # 同じ拘束を保てるよう、円筒の軸点・軸方向・半径そのものを
        # (単なる1点+向きの anchor とは別に) ここで解決しておく。
        # ui/viewport.py の _update_annotation_drag が読む。
        #
        # ユーザー指示（2026-09-27）: Diameter の "Show Annotation Anchors"
        # （赤ドット）は必ず円筒の中心軸上に来なければならない -- resolve_
        # anchor/_face_anchor 自体は（他のあらゆる面ベース特性と同じく）
        # 円筒面上の点を返すため、そこだけでは軸上に置けない。ui/viewport.py
        # がこの同じ axis_point/axis_dir を使って、表示中の anchor 点をその
        # 軸上（同じ軸方向位置）へ射影し直す。
        cylinder_axis = None
        if nominal.type_name in ("Radius", "Diameter"):
            cylinder_axis = resolve_cylinder_axis(nominal.feature_nominal_ids, doc)

        sub_anchor_recipe = None
        if nominal.type_name in _SUB_ANCHOR_TYPES:
            sub_anchor_recipe = resolve_sub_anchor_recipe(nominal.feature_nominal_ids, doc)

        # The bounded set of on-face positions "Balloon" style's automatic
        # anchor placement may move this characteristic's leader anchor to
        # -- see annotations/anchors.py's resolve_anchor_candidates. Empty
        # (feature not eligible for auto placement) for a target that
        # resolves to neither a plane nor a cylinder.
        anchor_candidates = resolve_anchor_candidates(nominal.feature_nominal_ids, doc, anchor[0])

        # "N×" pattern-count notation -- how many distinct instances this
        # one nominal's <FeatureNominalIds> ultimately resolves to, for a
        # characteristic that tolerances N instances of a feature directly
        # in one triad (e.g. one Flatness or Position applied to N
        # same-spec surfaces/holes at once) -- separate from the OTHER "N×"
        # case below (annotation_by_nominal_id loop further down), which
        # instead folds N *separate* single-feature nominals that happen to
        # share a value into one display.
        #
        # Two different counting bases, depending on what the characteristic
        # actually controls:
        #
        # 1. has_symbol types (Position/Flatness/SurfaceProfile/...) control
        #    actual B-Rep SURFACES directly, so the right count is resolved
        #    faces (_union_face_ids/len(face_ids)), not the raw
        #    feature_nominal_ids count: a single FeatureNominalIds entry can
        #    itself expand to more than one face -- either a "Group"
        #    FeatureNominal wrapping several sub-features (e.g. a "4 outer
        #    faces" group referenced by one FeatureNominalIds entry), or an
        #    ordinary Plane/CylinderFeatureNominal whose own EntityInternalIds
        #    lists more than one B-Rep entity (e.g. a single "height" feature
        #    nominal that is itself the pair of opposing planes it spans --
        #    confirmed against this app's own V-Block sample, whose
        #    Designator 04 profile characteristic has exactly one
        #    FeatureNominalIds entry but that entry's feature carries
        #    EntityInternalIds n="2"). Plain len(feature_nominal_ids) would
        #    miss both of these and wrongly show no "N×" at all.
        #
        # 2. "dimension" types (Width/Diameter/Radius/Length) measure a
        #    scalar SIZE property of a feature, not its faces directly, so
        #    face-counting is the wrong basis for them:
        #    - Width/Length are inherently measured *between* two (possibly
        #      different) features (e.g. two parallel planes), so one span
        #      always resolves 2 faces -- N repeated spans sharing one
        #      characteristic (e.g. 3 slots' width all toleranced by a
        #      single shared value) therefore resolve 2N faces, not N.
        #      Showing that raw face count as "2N×" would overstate the
        #      pattern by 2x, so it's halved back down to the true span
        #      count (2026-08-22, user-directed: "6x is a pure face count;
        #      for Width two faces calibrate one width, so the count is
        #      half, 3x"). An odd resolved count breaks the span-pairing
        #      assumption (malformed/unexpected data) -- warn and show no
        #      prefix rather than a wrong number.
        #    - Diameter/Radius are each inherently a property of ONE
        #      feature, so the right count is the number of FEATURE
        #      instances referenced (count_feature_instances -- expanding
        #      through a "Group" wrapper same as face-counting does, but
        #      never through a single leaf feature's own EntityInternalIds),
        #      not resolved faces: a hole's cylindrical wall commonly
        #      resolves to more than one B-Rep patch without that meaning
        #      more than one hole (2026-08-22, user-directed bug fix: an
        #      L-Bracket Diameter nominal with 4 FeatureNominalIds entries,
        #      each one hole resolving to 2 faces, was showing "8×∅3.5"
        #      instead of the correct "4×∅3.5" -- confirmed against that
        #      QIF file's own author-provided _3dv.FeatureCounter="4" /
        #      _3dv.TextPrefix="4X" attributes).
        face_ids = _union_face_ids(nominal.feature_nominal_ids, doc)
        dimension_face_opts = (
            dimension_face_options(face_ids, doc, size_endpoints)
            if size_endpoints is not None else []
        )
        if nominal.type_name in ("Width", "Length"):
            if len(face_ids) % 2 != 0:
                doc.warn(
                    f"CharacteristicNominal {nominal.id} ({nominal.type_name}): odd resolved face "
                    f"count ({len(face_ids)}); can't derive a span-pattern count, showing none"
                )
                feature_count = 1
            else:
                feature_count = len(face_ids) // 2
        elif spec.symbol_kind == "dimension":
            feature_count = _total_feature_instances(nominal.feature_nominal_ids, doc)
        else:
            feature_count = len(face_ids)
        count = feature_count if feature_count > 1 else 1

        line_range = doc.source_lines.get(nominal.id)
        annotation_by_nominal_id[nominal.id] = Annotation(
            id=f"char-{nominal.id}", layer=nominal.type_name, label=label,
            anchor_point=anchor[0], anchor_direction=anchor[1], color=spec.color,
            measured=item.measured if item is not None else None,
            designator=nominal.designator,
            symbol_kind=spec.symbol_kind,
            value_text=_iso_value_text(definition, nominal),
            datum_refs=datum_refs,
            dimension_icon=spec.dimension_icon,
            count=count,
            size_endpoints=size_endpoints,
            size_end_normal_sign=size_end_normal_sign,
            size_nominal_value=nominal_value,
            dimension_face_options=dimension_face_opts,
            cylinder_axis=cylinder_axis,
            sub_anchor_recipe=sub_anchor_recipe,
            anchor_candidates=anchor_candidates,
            source_line_start=line_range[0] if line_range else None,
            source_line_end=line_range[1] if line_range else None,
            face_ids=face_ids,
        )

    # Group by the exact (sorted) set of *resolved B-Rep faces* toleranced
    # -- in file order, so the stacked widget lists them the same order
    # the QIF file does -- and fold every member after the first into
    # that first one's combined_annotations instead of adding it to the
    # top-level list. Keyed on face_ids (already resolved above), not the
    # raw feature_nominal_ids: two characteristics can legitimately
    # describe the exact same physical feature(s) via structurally
    # different FeatureNominal references -- e.g. a 4-hole pattern's
    # Diameter listing the 4 CylinderFeatureNominals directly while its
    # Position instead references one "Group" FeatureNominal wrapping the
    # same 4 -- so comparing the feature_nominal_ids themselves would
    # miss that they're the same feature.
    feature_key_to_nominal_ids: dict[tuple[str, ...], list[str]] = {}
    for nominal in doc.characteristic_nominals.values():
        if nominal.id not in annotation_by_nominal_id:
            continue
        if nominal.type_name in _NEVER_COMBINE_TYPES:
            continue
        key = tuple(sorted(annotation_by_nominal_id[nominal.id].face_ids))
        if key:
            feature_key_to_nominal_ids.setdefault(key, []).append(nominal.id)

    folded_nominal_ids: set[str] = set()
    for nominal_ids in feature_key_to_nominal_ids.values():
        if len(nominal_ids) < 2:
            continue
        primary = annotation_by_nominal_id[nominal_ids[0]]
        for nominal_id in nominal_ids[1:]:
            member = annotation_by_nominal_id[nominal_id]
            # A folded member shares this exact (sorted) face_ids set with
            # the primary by construction (that's the join key above), so
            # any "N×" its own count would draw describes the exact same
            # group of faces the primary's own "N×" already labels once --
            # e.g. a Position combined onto a Width that spans the same
            # 3-slot, 6-face group independently resolves count=6 (Position
            # isn't a paired-span type, so it isn't halved the way Width
            # is), which would show a redundant, inconsistent "6×" right
            # next to the primary's already-correct "3×" for the identical
            # group (2026-08-22, user-directed: confirmed "3×" fixed, "the
            # 6× is unnecessary, delete it"). Suppress it on every folded
            # member -- the primary alone carries the group's cardinality.
            member.count = 1
            primary.combined_annotations.append(member)
            folded_nominal_ids.add(nominal_id)

    # A hole (or other feature) pattern is commonly toleranced by ONE
    # geometric characteristic covering the whole pattern (e.g. a Position
    # tolerance with a common zone across 3 holes) while each individual
    # hole still gets its own separate size characteristic (one Diameter
    # per hole, same nominal value) -- these don't share the exact same
    # face_ids (each Diameter only covers its own hole, not the whole
    # pattern), so the exact-match combining above leaves them as 3
    # separate, repeated "⌀3.5±0.1" callouts. ISO 1101/ASME Y14.5 instead
    # shows this once, as "3×⌀3.5±0.1", stacked above the pattern's shared
    # tolerance frame -- so: for each still-standalone symbol-bearing
    # characteristic, fold in any still-standalone "dimension" (Width/
    # Diameter/Radius/Length) characteristics whose own face_ids are
    # entirely covered by it and which share an identical value, replacing
    # all of them with the first one, tagged with how many it stands for.
    for nominal in doc.characteristic_nominals.values():
        if nominal.id in folded_nominal_ids or nominal.id not in annotation_by_nominal_id:
            continue
        frame = annotation_by_nominal_id[nominal.id]
        if not _has_symbol(frame.symbol_kind) or not frame.face_ids:
            continue
        frame_faces = set(frame.face_ids)
        members_by_value: dict[tuple[str, str], list[str]] = {}
        for other in doc.characteristic_nominals.values():
            if (
                other.id in folded_nominal_ids
                or other.id not in annotation_by_nominal_id
                or other.id == nominal.id
            ):
                continue
            dim = annotation_by_nominal_id[other.id]
            if dim.symbol_kind != "dimension" or not dim.face_ids:
                continue
            if not set(dim.face_ids) <= frame_faces:
                continue
            key = (dim.value_text or "", dim.dimension_icon)
            members_by_value.setdefault(key, []).append(other.id)

        for group_nominal_ids in members_by_value.values():
            if len(group_nominal_ids) < 2:
                continue
            primary = annotation_by_nominal_id[group_nominal_ids[0]]
            primary.count = len(group_nominal_ids)
            frame.combined_annotations.append(primary)
            for group_nominal_id in group_nominal_ids:
                folded_nominal_ids.add(group_nominal_id)
            # frame's own count (resolved independently, from its own
            # face_ids, earlier in this function) is now a redundant --
            # and sometimes actively wrong -- second "N×" for the exact
            # same pattern primary's own count just established: e.g. an
            # L-Bracket Position spanning 6 resolved faces (3 holes, each
            # counted twice because each hole's own wall resolves to 2
            # B-Rep faces) combined with its 3 same-value Diameter members
            # was showing a spurious "6×" right next to the Diameter's own
            # correct "3×" (2026-08-22, user-directed: "designator24's 6x
            # -- delete it, the size's ∅3.5 already has 3x, so it's
            # unnecessary"). The now-combined dimension's own count is the
            # authoritative pattern count for this group; suppress the
            # frame's to avoid ever showing two different numbers for the
            # same pattern.
            frame.count = 1

    for nominal_id, ann in annotation_by_nominal_id.items():
        if nominal_id not in folded_nominal_ids:
            annotations.append(ann)

    # A datum feature that's *also* directly toleranced by some
    # characteristic gets its datum-feature symbol attached below that
    # characteristic's own frame (Annotation.attached_datum_labels)
    # instead of a separate, freestanding datum annotation. Run this
    # *after* the combining above (not folded into the per-nominal loop
    # that builds annotation_by_nominal_id) so that, among several
    # characteristics combined onto the same feature, the datum lands on
    # whichever one actually has an ISO symbol frame to hang the triangle
    # off of (e.g. a Perpendicularity FCF) rather than arbitrarily
    # whichever happened to come first in the file (which might be a
    # plain-text Width dimension with no frame to attach to at all).
    # Keyed on resolved B-Rep faces, not raw feature_nominal_ids, for the
    # same reason the combining above is: a datum and a characteristic can
    # reference the exact same physical feature via structurally
    # different FeatureNominals.
    feature_to_nominal_ids: dict[str, list[str]] = {}
    for nominal in doc.characteristic_nominals.values():
        if nominal.id not in annotation_by_nominal_id:
            continue
        for fid in annotation_by_nominal_id[nominal.id].face_ids:
            feature_to_nominal_ids.setdefault(fid, []).append(nominal.id)

    consumed_datum_ids: set[str] = set()
    for datum in doc.datums.values():
        candidate_ids: list[str] = []
        seen_candidates: set[str] = set()
        for fid in _union_face_ids(datum.feature_nominal_ids, doc):
            for nid in feature_to_nominal_ids.get(fid, []):
                if nid not in seen_candidates:
                    seen_candidates.add(nid)
                    candidate_ids.append(nid)
        if not candidate_ids:
            continue
        target_id = next(
            (nid for nid in candidate_ids if _has_symbol(annotation_by_nominal_id[nid].symbol_kind)),
            candidate_ids[0],
        )
        annotation_by_nominal_id[target_id].attached_datum_labels.append(datum.label)
        consumed_datum_ids.add(datum.id)

    for datum in doc.datums.values():
        if datum.id in consumed_datum_ids:
            continue
        anchor = _average_anchor(datum.feature_nominal_ids, doc)
        if anchor is None:
            doc.warn(f"Datum {datum.label} ({datum.id}): no resolvable anchor; skipped")
            continue
        line_range = doc.source_lines.get(datum.id)
        annotations.append(Annotation(
            id=f"datum-{datum.id}", layer="Datums", label=datum.label,
            anchor_point=anchor[0], anchor_direction=anchor[1], color=_DATUM_COLOR,
            source_line_start=line_range[0] if line_range else None,
            source_line_end=line_range[1] if line_range else None,
            face_ids=_union_face_ids(datum.feature_nominal_ids, doc),
        ))

    return annotations
