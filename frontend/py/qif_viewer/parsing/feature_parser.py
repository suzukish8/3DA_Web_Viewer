"""Parses Features/FeatureNominals into model.features dataclasses.

Two well-understood types (Plane, Cylinder) get precise dataclasses; anything
else (Group, OppositeParallelPlanes, OtherSurface, and any future type QIF
adds) becomes a GenericFeatureNominal with best-effort anchor extraction --
CenterPlane/Origin/Axis fields if present, otherwise None (resolved later,
recursively for composite Group features, in annotations/anchors.py).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from qif_viewer.model.document import QifDocument
from qif_viewer.model.features import CylinderFeatureNominal, GenericFeatureNominal, PlaneFeatureNominal

from .xml_utils import find, id_list, local_tag, parse_point, text_of


def _parse_plane(el: ET.Element) -> PlaneFeatureNominal | None:
    fid = el.get("id")
    location = parse_point(el, "Location")
    normal = parse_point(el, "Normal")
    if fid is None or location is None or normal is None:
        return None
    return PlaneFeatureNominal(
        id=fid,
        name=text_of(el, "Name") or "",
        feature_definition_id=text_of(el, "FeatureDefinitionId") or "",
        location=location,
        normal=normal,
        entity_internal_ids=id_list(el, "EntityInternalIds"),
    )


def _parse_cylinder(el: ET.Element) -> CylinderFeatureNominal | None:
    fid = el.get("id")
    axis = find(el, "Axis")
    if fid is None or axis is None:
        return None
    axis_point = parse_point(axis, "AxisPoint")
    axis_dir = parse_point(axis, "Direction")
    if axis_point is None or axis_dir is None:
        return None
    return CylinderFeatureNominal(
        id=fid,
        name=text_of(el, "Name") or "",
        feature_definition_id=text_of(el, "FeatureDefinitionId") or "",
        axis_point=axis_point,
        axis_dir=axis_dir,
        entity_internal_ids=id_list(el, "EntityInternalIds"),
    )


def _parse_generic(el: ET.Element, kind: str) -> GenericFeatureNominal | None:
    fid = el.get("id")
    if fid is None:
        return None
    # Best-effort anchor: try the shapes we've actually seen in the wild.
    location = parse_point(el, "Location")
    if location is None:
        location = parse_point(el, "CenterPlane/Point")
    if location is None:
        location = parse_point(el, "Origin")
    normal = parse_point(el, "Normal")
    if normal is None:
        normal = parse_point(el, "CenterPlane/Normal")
    axis = find(el, "Axis")
    axis_dir = None
    if axis is not None:
        axis_point = parse_point(axis, "AxisPoint")
        axis_dir = parse_point(axis, "Direction")
        if location is None:
            location = axis_point
    if location is None and normal is not None:
        location = None  # normal without a point isn't a usable anchor
    return GenericFeatureNominal(
        id=fid,
        name=text_of(el, "Name") or "",
        kind=kind,
        location=location,
        axis_dir=axis_dir if axis_dir is not None else normal,
        raw_element=el,
        entity_internal_ids=id_list(el, "EntityInternalIds"),
    )


_PARSERS = {
    "PlaneFeatureNominal": _parse_plane,
    "CylinderFeatureNominal": _parse_cylinder,
}


def parse_features(root: ET.Element, doc: QifDocument) -> None:
    features = find(root, "Features")
    if features is None:
        return
    nominals = find(features, "FeatureNominals")
    if nominals is None:
        return
    for el in list(nominals):
        tag = local_tag(el)
        parser = _PARSERS.get(tag)
        if parser is not None:
            parsed = parser(el)
            if parsed is None:
                doc.warn(f"{tag} {el.get('id')} missing required fields; skipped")
                continue
            doc.feature_nominals[parsed.id] = parsed
        else:
            kind = tag.replace("FeatureNominal", "") or tag
            parsed = _parse_generic(el, kind)
            if parsed is None:
                doc.warn(f"{tag} {el.get('id')} could not be parsed; skipped")
                continue
            if parsed.location is None and parsed.kind != "Group":
                doc.warn(f"{tag} {parsed.id} has no direct anchor; best-effort resolution only")
            doc.feature_nominals[parsed.id] = parsed
