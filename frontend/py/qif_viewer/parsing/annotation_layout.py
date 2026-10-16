"""Persists ui/viewport.py's per-annotation label/anchor/dimension-line/
dimension-endpoint positions (wherever the user has dragged them), the
active annotation
display style ("billboard" or "plane"), any per-annotation plane, up-axis,
or dimension-leader-end overrides (which principal plane a "plane"-style
annotation is forced flat against, which signed world axis direction is
forced "up" within it, and -- for a symbol-bearing ISO GPS frame combined
with its own size dimension -- which end of that dimension line its
shared leader starts from; which edge of the label its leader line
connects to is no longer an overridable choice at all -- see ui/
viewport.py's _pick_leader_attach_side), a size dimension's own
extension-line direction override
(which of its two in-plane perpendicular candidates, and which sign, its
witness lines project along), whether its extension lines are shown at
all (off collapses them to zero length, running the dimension line
straight through the measured points), and whether its own arrowheads
point inward or outward, which annotations are currently hidden via the
sidebar's checkboxes, the live camera's current view, the Annotation Size
slider value, whether the tolerance-frame/datum-box's own backing quad is
painted white, and how far each of the three Hidden-Side Mirrors sits
from the model, to a sidecar ``<basename>.ant`` XML file next to the QIF
file, rather than into the QIF file's own text -- these are viewport-only
presentation choices, not part of the inspection data itself.

Kept in QIF's own naming idiom (PascalCase elements, "id"/"n" attributes,
a versioned root) on purpose, so this data could plausibly be folded into
the QIF file itself later as a vendor-extension block, the same way saved
camera views are (see parsing/saved_views.py) -- but unlike saved_views.py
this is a brand-new standalone file, not an edit into an existing
document, so there's no reason to avoid a normal DOM-based read/write here.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

_ROOT_TAG = "QIFViewerAnnotationLayout"
_VERSION = "1.2"

_VALID_STYLES = ("billboard", "plane", "balloon", "none", "graphic_pmi")
_VALID_PLANES = ("XY", "YZ", "ZX")
_VALID_UP_KEYS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
_VALID_AXES = ("X", "Y", "Z")
_VALID_EXT_DIR_CHOICES = ("a+", "a-", "b+", "b-")
_VALID_DIMENSION_LEADER_ENDS = ("d1", "d2")


def _default_layout() -> dict:
    return {
        "style": None,
        "labels": {},
        "label_positions_are_attach_points": False,
        "anchors": {},
        "dimension_lines": {},
        "dimension_endpoints": {},
        "dimension_ext_dir_overrides": {},
        "dimension_extension_disabled": [],
        "dimension_arrows_outward": [],
        "dimension_anchor_slides": {},
        "dimension_face_choices": {},
        "anchor_sections": [],
        "dimension_ext_start_choices": {},
        "plane_overrides": {},
        "up_overrides": {},
        "dimension_leader_end_overrides": {},
        "camera_view": None,
        "annotation_scale": None,
        "annotation_frame_height_mm": None,
        "dot_scale": None,
        "dot_size_mm": None,
        "arrow_size_mm": None,
        "hidden_ids": [],
        "balloon_min_leader_angle": None,
        "annotation_backing_fill": None,
        "mirror_offsets": {},
        "datum_systems_visible": [],
        "theoretical_exact_dimensions_visible": None,
    }


def _escape_xml_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _format_vec(v) -> str:
    return " ".join(repr(float(c)) for c in v)


def _parse_vec(text: Optional[str]) -> Optional[tuple]:
    if text is None:
        return None
    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def read_annotation_layout(xml_text: str) -> dict:
    """Returns _default_layout() (style None, everything else empty) if
    the text isn't parseable XML, isn't our root element, or has no usable
    entries -- a missing/foreign/hand-edited .ant file should never crash
    the app, just behave as if it weren't there."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return _default_layout()
    if root.tag != _ROOT_TAG:
        return _default_layout()

    style_text = root.findtext("Style")
    style = style_text.strip() if style_text is not None and style_text.strip() in _VALID_STYLES else None

    labels: dict = {}
    for el in root.findall("./Labels/Label"):
        annotation_id = el.get("id")
        position = _parse_vec(el.findtext("Position"))
        if annotation_id and position is not None:
            labels[annotation_id] = position

    # Distinguishes a current-format file (each Label/Position above is
    # ui/viewport.py's own managed reference point -- the leader's bend
    # point as of 2026-08-11, its edge before that) from an older one
    # (saved before that became the managed coordinate, where it was the
    # box's own rendered center instead) -- see ui/viewport.py's
    # apply_annotation_layout, which needs this to convert an old file's
    # positions without visibly moving anything. Absent entirely (any
    # file predating this element, or a hand-edited one) defaults to
    # False, i.e. the older, center-based format. The element name itself
    # (LabelPositionsAreAttachPoints) predates the bend-point change and
    # is kept as-is for on-disk compatibility -- it still means exactly
    # what it always did, "not the older center-based format".
    label_positions_are_attach_points = (root.findtext("LabelPositionsAreAttachPoints") or "").strip() == "true"

    anchors: dict = {}
    for el in root.findall("./Anchors/Anchor"):
        annotation_id = el.get("id")
        position = _parse_vec(el.findtext("Position"))
        direction = _parse_vec(el.findtext("Direction"))
        if annotation_id and position is not None and direction is not None:
            anchors[annotation_id] = {"position": position, "direction": direction}

    dimension_lines: dict = {}
    for el in root.findall("./DimensionLines/DimensionLine"):
        part_id = el.get("id")
        text = el.findtext("ExtensionLength")
        if part_id and text is not None:
            try:
                dimension_lines[part_id] = float(text)
            except ValueError:
                pass

    dimension_endpoints: dict = {}
    for el in root.findall("./DimensionEndpoints/DimensionEndpoint"):
        part_id = el.get("id")
        if not part_id:
            continue
        entry: dict = {}
        p1 = _parse_vec(el.findtext("P1"))
        if p1 is not None:
            entry["p1"] = p1
        p2 = _parse_vec(el.findtext("P2"))
        if p2 is not None:
            entry["p2"] = p2
        if entry:
            dimension_endpoints[part_id] = entry

    dimension_ext_dir_overrides: dict = {}
    for el in root.findall("./DimensionExtDirOverrides/DimensionExtDirOverride"):
        part_id = el.get("id")
        choice = el.get("dir")
        if part_id and choice in _VALID_EXT_DIR_CHOICES:
            dimension_ext_dir_overrides[part_id] = choice

    dimension_extension_disabled: list = []
    for el in root.findall("./DimensionExtensionDisabled/Disabled"):
        part_id = el.get("id")
        if part_id:
            dimension_extension_disabled.append(part_id)

    dimension_arrows_outward: list = []
    for el in root.findall("./DimensionArrowsOutward/Outward"):
        part_id = el.get("id")
        if part_id:
            dimension_arrows_outward.append(part_id)

    # Manual Anchor Placement of a dimension-anchored characteristic:
    # owner annotation id -> 1-based on-face candidate index its whole
    # dimension line was slid onto (ui/viewport.py's _dimension_anchor_
    # slide_choice / set_dimension_anchor_slide_choice).
    dimension_anchor_slides: dict = {}
    for el in root.findall("./DimensionAnchorSlides/DimensionAnchorSlide"):
        annotation_id = el.get("id")
        index_text = el.get("index")
        if not annotation_id or index_text is None:
            continue
        try:
            index = int(index_text)
        except ValueError:
            continue
        if index >= 1:
            dimension_anchor_slides[annotation_id] = index

    # Right-click "Set Dimension Face..." -- part id -> 0-based option index.
    dimension_face_choices: dict = {}
    for el in root.findall("./DimensionFaceChoices/DimensionFaceChoice"):
        part_id = el.get("id")
        index_text = el.get("index")
        if not part_id or index_text is None:
            continue
        try:
            index = int(index_text)
        except ValueError:
            continue
        if index >= 0:
            dimension_face_choices[part_id] = index

    # Right-click "Dimension Extension Start > (a sub-anchor point)" --
    # part id -> 0-based sub-anchor index the size dimension's extension
    # line was slid to start from (ui/viewport.py's _dimension_ext_start_
    # choice / set_dimension_ext_start_choice). Balloon & Mirror only.
    dimension_ext_start_choices: dict = {}
    for el in root.findall("./DimensionExtStartChoices/DimensionExtStartChoice"):
        part_id = el.get("id")
        index_text = el.get("index")
        if not part_id or index_text is None:
            continue
        try:
            index = int(index_text)
        except ValueError:
            continue
        if index >= 0:
            dimension_ext_start_choices[part_id] = index

    mirror_offsets: dict = {}
    for el in root.findall("./MirrorOffsets/MirrorOffset"):
        axis = el.get("axis")
        text = el.text
        if axis in _VALID_AXES and text is not None:
            try:
                mirror_offsets[axis] = float(text)
            except ValueError:
                pass

    plane_overrides: dict = {}
    for el in root.findall("./PlaneOverrides/PlaneOverride"):
        annotation_id = el.get("id")
        plane = el.get("plane")
        if annotation_id and plane in _VALID_PLANES:
            plane_overrides[annotation_id] = plane

    up_overrides: dict = {}
    for el in root.findall("./UpOverrides/UpOverride"):
        annotation_id = el.get("id")
        axis = el.get("axis")
        if annotation_id and axis in _VALID_UP_KEYS:
            up_overrides[annotation_id] = axis

    dimension_leader_end_overrides: dict = {}
    for el in root.findall("./DimensionLeaderEndOverrides/DimensionLeaderEndOverride"):
        annotation_id = el.get("id")
        end = el.get("end")
        if annotation_id and end in _VALID_DIMENSION_LEADER_ENDS:
            dimension_leader_end_overrides[annotation_id] = end

    camera_view: Optional[dict] = None
    camera_el = root.find("./CameraView")
    if camera_el is not None:
        position = _parse_vec(camera_el.findtext("Position"))
        focal_point = _parse_vec(camera_el.findtext("FocalPoint"))
        view_up = _parse_vec(camera_el.findtext("ViewUp"))
        scale_text = camera_el.findtext("ParallelScale")
        parallel_scale = None
        if scale_text is not None:
            try:
                parallel_scale = float(scale_text)
            except ValueError:
                parallel_scale = None
        if position is not None and focal_point is not None and view_up is not None and parallel_scale is not None:
            camera_view = {
                "position": position,
                "focal_point": focal_point,
                "view_up": view_up,
                "parallel_scale": parallel_scale,
            }

    annotation_scale: Optional[float] = None
    scale_text = root.findtext("AnnotationScale")
    if scale_text is not None:
        try:
            annotation_scale = float(scale_text)
        except ValueError:
            annotation_scale = None

    # FCF Height -- the feature control frame's own height, in the model's
    # own units (mm). Preferred over the legacy AnnotationScale (a unitless
    # multiple of an internal bbox-relative default): a real length is
    # meaningful on its own, independent of this part's bounding box. When
    # both are present this one wins; when only AnnotationScale is present
    # the caller converts it through the loaded model's bbox.
    annotation_frame_height_mm: Optional[float] = None
    height_text = root.findtext("AnnotationFrameHeightMM")
    if height_text is not None:
        try:
            annotation_frame_height_mm = float(height_text)
        except ValueError:
            annotation_frame_height_mm = None

    dot_scale: Optional[float] = None
    dot_scale_text = root.findtext("DotScale")
    if dot_scale_text is not None:
        try:
            dot_scale = float(dot_scale_text)
        except ValueError:
            dot_scale = None

    # Dot Size -- the anchor-end dot's diameter in the model's own units
    # (mm). Same story as AnnotationFrameHeightMM above vs AnnotationScale:
    # preferred over the legacy unitless DotScale; when only DotScale is
    # present the caller converts it through the loaded model's bbox.
    dot_size_mm: Optional[float] = None
    dot_size_text = root.findtext("DotSizeMM")
    if dot_size_text is not None:
        try:
            dot_size_mm = float(dot_size_text)
        except ValueError:
            dot_size_mm = None

    # Terminator Arrow Size (2026-10-13) -- flat arrowhead apex-to-base
    # length in mm; split from the dot size above.
    arrow_size_mm: Optional[float] = None
    arrow_size_text = root.findtext("ArrowSizeMM")
    if arrow_size_text is not None:
        try:
            arrow_size_mm = float(arrow_size_text)
        except ValueError:
            arrow_size_mm = None

    hidden_ids: list = []
    for el in root.findall("./HiddenAnnotations/Hidden"):
        annotation_id = el.get("id")
        if annotation_id:
            hidden_ids.append(annotation_id)

    balloon_min_leader_angle: Optional[float] = None
    angle_text = root.findtext("BalloonMinLeaderAngle")
    if angle_text is not None:
        try:
            balloon_min_leader_angle = float(angle_text)
        except ValueError:
            balloon_min_leader_angle = None

    annotation_backing_fill: Optional[bool] = None
    fill_text = root.findtext("AnnotationBackingFill")
    if fill_text is not None and fill_text.strip() in ("true", "false"):
        annotation_backing_fill = fill_text.strip() == "true"

    # View > Datum Systems -- which DatumReferenceFrame ids the user turned
    # on (ui/viewport.py's _datum_system_visible). Ids are per-file (a
    # DRF's own <DatumReferenceFrame id="..."> from the QIF source), so a
    # stale entry left over from a since-edited file just resolves to no
    # menu item and is silently dropped -- same "never crash on foreign/
    # outdated data" treatment as everything else in this file.
    datum_systems_visible: list = []
    for el in root.findall("./DatumSystemsVisible/Visible"):
        drf_id = el.get("id")
        if drf_id:
            datum_systems_visible.append(drf_id)

    theoretical_exact_dimensions_visible: Optional[bool] = None
    ted_text = root.findtext("TheoreticalExactDimensionsVisible")
    if ted_text is not None and ted_text.strip() in ("true", "false"):
        theoretical_exact_dimensions_visible = ted_text.strip() == "true"

    # View > Section View -- registered sections (ui/section_panel.py's
    # capture_state() plus the "Show Gray Outline" checkbox).
    sections: list = []
    for el in root.findall("./Sections/Section"):
        name = el.get("name")
        if not name:
            continue
        try:
            position = float(el.get("position", "0"))
            hatch_angle = float(el.get("hatch_angle", "45"))
            hatch_spacing = float(el.get("hatch_spacing", "1"))
            hatch_band_width = float(el.get("hatch_band_width", "5"))
            hatch_width = float(el.get("hatch_width", "1"))
            color = [float(c) for c in (el.get("color") or "0.55 0.55 0.55").split()][:3]
            if len(color) != 3:
                color = [0.55, 0.55, 0.55]
            normal = _parse_vec(el.findtext("CustomNormal")) or (0.0, 0.0, 1.0)
        except ValueError:
            continue
        sections.append({
            "name": name,
            "state": {
                "orientation_text": el.get("orientation") or "XY Plane (normal Z)",
                "custom_normal": list(normal),
                "position": position,
                "flip": el.get("flip") == "true",
                "hatch": el.get("hatch") != "false",
                "hatch_angle": hatch_angle,
                "hatch_spacing": hatch_spacing,
                "hatch_band": el.get("hatch_band") != "false",
                "hatch_band_width": hatch_band_width,
                "hatch_width": hatch_width,
                "color": color,
            },
            "outline_visible": el.get("outline") != "false",
        })

    anchor_sections: list = []
    for el in root.findall("./AnchorSections/AnchorSection"):
        try:
            radius_text = el.get("radius")
            origin = _parse_vec(el.findtext("Origin"))
            plane = el.get("plane")
            ann_id = el.get("id")
            color = [float(c) for c in (el.get("color") or "0.55 0.55 0.55").split()][:3]
            if not origin or not ann_id or plane not in ("XY", "YZ", "ZX") or len(color) != 3:
                continue
            anchor_sections.append({
                "annotation_id": ann_id, "plane": plane, "origin": list(origin),
                "radius": float(radius_text) if radius_text else None,
                "color": color, "hatch_width": float(el.get("hatch_width", "1")),
            })
        except ValueError:
            continue

    return {
        "anchor_sections": anchor_sections,
        "sections": sections,
        "style": style,
        "labels": labels,
        "label_positions_are_attach_points": label_positions_are_attach_points,
        "anchors": anchors,
        "dimension_lines": dimension_lines,
        "dimension_endpoints": dimension_endpoints,
        "dimension_ext_dir_overrides": dimension_ext_dir_overrides,
        "dimension_extension_disabled": dimension_extension_disabled,
        "dimension_arrows_outward": dimension_arrows_outward,
        "dimension_anchor_slides": dimension_anchor_slides,
        "dimension_face_choices": dimension_face_choices,
        "dimension_ext_start_choices": dimension_ext_start_choices,
        "plane_overrides": plane_overrides,
        "up_overrides": up_overrides,
        "dimension_leader_end_overrides": dimension_leader_end_overrides,
        "camera_view": camera_view,
        "annotation_scale": annotation_scale,
        "annotation_frame_height_mm": annotation_frame_height_mm,
        "dot_scale": dot_scale,
        "dot_size_mm": dot_size_mm,
        "arrow_size_mm": arrow_size_mm,
        "hidden_ids": hidden_ids,
        "balloon_min_leader_angle": balloon_min_leader_angle,
        "annotation_backing_fill": annotation_backing_fill,
        "mirror_offsets": mirror_offsets,
        "datum_systems_visible": datum_systems_visible,
        "theoretical_exact_dimensions_visible": theoretical_exact_dimensions_visible,
    }


def write_annotation_layout(layout: dict) -> str:
    """`layout` shape matches read_annotation_layout()'s return value:
    {"style": "billboard"|"plane"|None, "labels": {id: (x,y,z)},
    "label_positions_are_attach_points": bool, "anchors":
    {id: {"position": (x,y,z), "direction": (x,y,z)}}, "dimension_lines":
    {id: ext_len}, "dimension_endpoints": {id: {"p1": (x,y,z)|omitted,
    "p2": (x,y,z)|omitted}}, "dimension_ext_dir_overrides":
    {id: "a+"|"a-"|"b+"|"b-"}, "dimension_extension_disabled": [id, ...],
    "dimension_arrows_outward": [id, ...],
    "dimension_anchor_slides": {owner_id: int >= 1},
    "dimension_face_choices": {id: int >= 0},
    "dimension_ext_start_choices": {id: int >= 0},
    "plane_overrides": {id: "XY"|"YZ"|"ZX"},
    "up_overrides": {id: "+X"|"-X"|"+Y"|"-Y"|"+Z"|"-Z"},
    "dimension_leader_end_overrides": {id: "d1"|"d2"}, "camera_view":
    {"position": (x,y,z), "focal_point": (x,y,z), "view_up": (x,y,z),
    "parallel_scale": float} | None, "annotation_scale": float | None,
    "annotation_frame_height_mm": float | None,
    "dot_scale": float | None, "dot_size_mm": float | None,
    "hidden_ids": [id, ...],
    "balloon_min_leader_angle": float | None,
    "annotation_backing_fill": bool | None,
    "mirror_offsets": {"X"|"Y"|"Z": float},
    "datum_systems_visible": [drf_id, ...],
    "theoretical_exact_dimensions_visible": bool | None}.
    Any key may be omitted (treated as empty/None)."""
    style = layout.get("style")
    labels = layout.get("labels") or {}
    label_positions_are_attach_points = bool(layout.get("label_positions_are_attach_points", False))
    anchors = layout.get("anchors") or {}
    dimension_lines = layout.get("dimension_lines") or {}
    dimension_endpoints = layout.get("dimension_endpoints") or {}
    dimension_ext_dir_overrides = layout.get("dimension_ext_dir_overrides") or {}
    dimension_extension_disabled = layout.get("dimension_extension_disabled") or []
    dimension_arrows_outward = layout.get("dimension_arrows_outward") or []
    dimension_anchor_slides = layout.get("dimension_anchor_slides") or {}
    dimension_face_choices = layout.get("dimension_face_choices") or {}
    dimension_ext_start_choices = layout.get("dimension_ext_start_choices") or {}
    plane_overrides = layout.get("plane_overrides") or {}
    up_overrides = layout.get("up_overrides") or {}
    dimension_leader_end_overrides = layout.get("dimension_leader_end_overrides") or {}
    camera_view = layout.get("camera_view")
    annotation_scale = layout.get("annotation_scale")
    annotation_frame_height_mm = layout.get("annotation_frame_height_mm")
    dot_scale = layout.get("dot_scale")
    dot_size_mm = layout.get("dot_size_mm")
    arrow_size_mm = layout.get("arrow_size_mm")
    hidden_ids = layout.get("hidden_ids") or []
    balloon_min_leader_angle = layout.get("balloon_min_leader_angle")
    annotation_backing_fill = layout.get("annotation_backing_fill")
    mirror_offsets = layout.get("mirror_offsets") or {}
    datum_systems_visible = layout.get("datum_systems_visible") or []
    theoretical_exact_dimensions_visible = layout.get("theoretical_exact_dimensions_visible")

    lines = ['<?xml version="1.0" encoding="utf-8"?>', f'<{_ROOT_TAG} version="{_VERSION}">']

    if style in _VALID_STYLES:
        lines.append(f"\t<Style>{style}</Style>")

    if annotation_scale is not None:
        lines.append(f"\t<AnnotationScale>{float(annotation_scale)!r}</AnnotationScale>")

    if annotation_frame_height_mm is not None:
        lines.append(
            f"\t<AnnotationFrameHeightMM>{float(annotation_frame_height_mm)!r}</AnnotationFrameHeightMM>"
        )

    if dot_scale is not None:
        lines.append(f"\t<DotScale>{float(dot_scale)!r}</DotScale>")

    if dot_size_mm is not None:
        lines.append(f"\t<DotSizeMM>{float(dot_size_mm)!r}</DotSizeMM>")
    if arrow_size_mm is not None:
        lines.append(f"\t<ArrowSizeMM>{float(arrow_size_mm)!r}</ArrowSizeMM>")

    if balloon_min_leader_angle is not None:
        lines.append(f"\t<BalloonMinLeaderAngle>{float(balloon_min_leader_angle)!r}</BalloonMinLeaderAngle>")

    if annotation_backing_fill is not None:
        lines.append(f"\t<AnnotationBackingFill>{'true' if annotation_backing_fill else 'false'}</AnnotationBackingFill>")

    lines.append(
        f"\t<LabelPositionsAreAttachPoints>"
        f"{'true' if label_positions_are_attach_points else 'false'}"
        f"</LabelPositionsAreAttachPoints>"
    )

    lines.append(f'\t<Labels n="{len(labels)}">')
    for annotation_id, position in labels.items():
        lines.append(
            f'\t\t<Label id="{_escape_xml_attr(annotation_id)}">'
            f"<Position>{_format_vec(position)}</Position></Label>"
        )
    lines.append("\t</Labels>")

    lines.append(f'\t<Anchors n="{len(anchors)}">')
    for annotation_id, data in anchors.items():
        lines.append(
            f'\t\t<Anchor id="{_escape_xml_attr(annotation_id)}">'
            f'<Position>{_format_vec(data["position"])}</Position>'
            f'<Direction>{_format_vec(data["direction"])}</Direction></Anchor>'
        )
    lines.append("\t</Anchors>")

    lines.append(f'\t<DimensionLines n="{len(dimension_lines)}">')
    for part_id, ext_len in dimension_lines.items():
        lines.append(
            f'\t\t<DimensionLine id="{_escape_xml_attr(part_id)}">'
            f"<ExtensionLength>{ext_len!r}</ExtensionLength></DimensionLine>"
        )
    lines.append("\t</DimensionLines>")

    lines.append(f'\t<DimensionEndpoints n="{len(dimension_endpoints)}">')
    for part_id, points in dimension_endpoints.items():
        inner = "".join(
            f"<{tag.upper()}>{_format_vec(points[key])}</{tag.upper()}>"
            for key, tag in (("p1", "P1"), ("p2", "P2"))
            if key in points
        )
        lines.append(f'\t\t<DimensionEndpoint id="{_escape_xml_attr(part_id)}">{inner}</DimensionEndpoint>')
    lines.append("\t</DimensionEndpoints>")

    lines.append(f'\t<DimensionExtDirOverrides n="{len(dimension_ext_dir_overrides)}">')
    for part_id, choice in dimension_ext_dir_overrides.items():
        if choice in _VALID_EXT_DIR_CHOICES:
            lines.append(
                f'\t\t<DimensionExtDirOverride id="{_escape_xml_attr(part_id)}" dir="{choice}" />'
            )
    lines.append("\t</DimensionExtDirOverrides>")

    lines.append(f'\t<DimensionExtensionDisabled n="{len(dimension_extension_disabled)}">')
    for part_id in dimension_extension_disabled:
        lines.append(f'\t\t<Disabled id="{_escape_xml_attr(part_id)}" />')
    lines.append("\t</DimensionExtensionDisabled>")

    lines.append(f'\t<DimensionArrowsOutward n="{len(dimension_arrows_outward)}">')
    for part_id in dimension_arrows_outward:
        lines.append(f'\t\t<Outward id="{_escape_xml_attr(part_id)}" />')
    lines.append("\t</DimensionArrowsOutward>")

    lines.append(f'\t<DimensionAnchorSlides n="{len(dimension_anchor_slides)}">')
    for annotation_id, index in dimension_anchor_slides.items():
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if index >= 1:
            lines.append(
                f'\t\t<DimensionAnchorSlide id="{_escape_xml_attr(annotation_id)}" index="{index}" />'
            )
    lines.append("\t</DimensionAnchorSlides>")

    lines.append(f'\t<DimensionFaceChoices n="{len(dimension_face_choices)}">')
    for part_id, index in dimension_face_choices.items():
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if index >= 0:
            lines.append(
                f'\t\t<DimensionFaceChoice id="{_escape_xml_attr(part_id)}" index="{index}" />'
            )
    lines.append("\t</DimensionFaceChoices>")

    lines.append(f'\t<DimensionExtStartChoices n="{len(dimension_ext_start_choices)}">')
    for part_id, index in dimension_ext_start_choices.items():
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if index >= 0:
            lines.append(
                f'\t\t<DimensionExtStartChoice id="{_escape_xml_attr(part_id)}" index="{index}" />'
            )
    lines.append("\t</DimensionExtStartChoices>")

    lines.append(f'\t<MirrorOffsets n="{len(mirror_offsets)}">')
    for axis, offset in mirror_offsets.items():
        if axis in _VALID_AXES:
            lines.append(f'\t\t<MirrorOffset axis="{axis}">{float(offset)!r}</MirrorOffset>')
    lines.append("\t</MirrorOffsets>")

    lines.append(f'\t<PlaneOverrides n="{len(plane_overrides)}">')
    for annotation_id, plane in plane_overrides.items():
        lines.append(
            f'\t\t<PlaneOverride id="{_escape_xml_attr(annotation_id)}" plane="{_escape_xml_attr(plane)}" />'
        )
    lines.append("\t</PlaneOverrides>")

    lines.append(f'\t<UpOverrides n="{len(up_overrides)}">')
    for annotation_id, axis in up_overrides.items():
        lines.append(f'\t\t<UpOverride id="{_escape_xml_attr(annotation_id)}" axis="{_escape_xml_attr(axis)}" />')
    lines.append("\t</UpOverrides>")

    lines.append(f'\t<DimensionLeaderEndOverrides n="{len(dimension_leader_end_overrides)}">')
    for annotation_id, end in dimension_leader_end_overrides.items():
        if end in _VALID_DIMENSION_LEADER_ENDS:
            lines.append(
                f'\t\t<DimensionLeaderEndOverride id="{_escape_xml_attr(annotation_id)}" end="{end}" />'
            )
    lines.append("\t</DimensionLeaderEndOverrides>")

    lines.append(f'\t<HiddenAnnotations n="{len(hidden_ids)}">')
    for annotation_id in hidden_ids:
        lines.append(f'\t\t<Hidden id="{_escape_xml_attr(annotation_id)}" />')
    lines.append("\t</HiddenAnnotations>")

    lines.append(f'\t<DatumSystemsVisible n="{len(datum_systems_visible)}">')
    for drf_id in datum_systems_visible:
        lines.append(f'\t\t<Visible id="{_escape_xml_attr(drf_id)}" />')
    lines.append("\t</DatumSystemsVisible>")

    if theoretical_exact_dimensions_visible is not None:
        lines.append(
            "\t<TheoreticalExactDimensionsVisible>"
            f"{'true' if theoretical_exact_dimensions_visible else 'false'}"
            "</TheoreticalExactDimensionsVisible>"
        )

    sections = layout.get("sections") or []
    lines.append(f'\t<Sections n="{len(sections)}">')
    for sec in sections:
        st = sec["state"]
        lines.append(
            f'\t\t<Section name="{_escape_xml_attr(sec["name"])}"'
            f' orientation="{_escape_xml_attr(st["orientation_text"])}"'
            f' position="{float(st["position"])!r}"'
            f' flip="{"true" if st.get("flip") else "false"}"'
            f' hatch="{"true" if st.get("hatch", True) else "false"}"'
            f' hatch_angle="{float(st.get("hatch_angle", 45.0))!r}"'
            f' hatch_spacing="{float(st.get("hatch_spacing", 1.0))!r}"'
            f' hatch_band="{"true" if st.get("hatch_band", True) else "false"}"'
            f' hatch_band_width="{float(st.get("hatch_band_width", 5.0))!r}"'
            f' hatch_width="{float(st.get("hatch_width", 1.0))!r}"'
            f' color="{_format_vec(st.get("color", (0.55, 0.55, 0.55)))}"'
            f' outline="{"true" if sec.get("outline_visible", True) else "false"}">'
            f"<CustomNormal>{_format_vec(st['custom_normal'])}</CustomNormal></Section>"
        )
    lines.append("\t</Sections>")

    anchor_sections = layout.get("anchor_sections") or []
    lines.append(f'	<AnchorSections n="{len(anchor_sections)}">')
    for a in anchor_sections:
        radius = a.get("radius")
        lines.append(
            f'		<AnchorSection id="{_escape_xml_attr(a["annotation_id"])}"'
            f' plane="{a["plane"]}"'
            + (f' radius="{float(radius)!r}"' if radius is not None else "")
            + f' hatch_width="{float(a.get("hatch_width", 1.0))!r}"'
            f' color="{_format_vec(a.get("color", (0.55, 0.55, 0.55)))}">'
            f"<Origin>{_format_vec(a['origin'])}</Origin></AnchorSection>"
        )
    lines.append("	</AnchorSections>")

    if camera_view is not None:
        lines.append("\t<CameraView>")
        lines.append(f"\t\t<Position>{_format_vec(camera_view['position'])}</Position>")
        lines.append(f"\t\t<FocalPoint>{_format_vec(camera_view['focal_point'])}</FocalPoint>")
        lines.append(f"\t\t<ViewUp>{_format_vec(camera_view['view_up'])}</ViewUp>")
        lines.append(f"\t\t<ParallelScale>{camera_view['parallel_scale']!r}</ParallelScale>")
        lines.append("\t</CameraView>")

    lines.append(f"</{_ROOT_TAG}>")
    return "\n".join(lines) + "\n"
