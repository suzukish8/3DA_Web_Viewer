"""Parses datums, datum reference frames, and ISO GPS characteristic triads.

The characteristic triad ({Type}CharacteristicDefinition/Nominal/Item) is
parsed *generically*: we regex-match the tag name to recover `type_name`
and pull the shared core fields every triad member has. This means a brand
new characteristic type (Circularity, Perpendicularity, ...) parses with
zero code changes here -- only its *presentation* (symbol/color/label, in
annotations/registry.py) needs registering.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from qif_viewer.model.document import QifDocument
from qif_viewer.model.gdt import (
    CharacteristicDefinition,
    CharacteristicItem,
    CharacteristicNominal,
    Datum,
    DatumReferenceFrame,
)

from .xml_utils import attribute_values, find, findall, id_list, local_tag, parse_float, text_of

_TRIAD_RE = re.compile(r"^(?P<type_name>\w+?)Characteristic(?P<role>Definition|Nominal|Item)$")

# Fields consumed generically -- anything else on a triad element is stashed
# into CharacteristicDefinition.raw_extra instead of being dropped.
_DEFINITION_CORE_TAGS = {
    "ToleranceValue", "DatumReferenceFrameId", "MaterialCondition", "ZoneShape", "CommonZone", "Attributes",
}


def parse_datums(root: ET.Element, doc: QifDocument) -> None:
    container = find(root, "DatumDefinitions")
    if container is None:
        return
    for el in findall(container, "DatumDefinition"):
        did = el.get("id")
        label = text_of(el, "DatumLabel")
        if did is None or label is None:
            doc.warn(f"DatumDefinition {did} missing id/label; skipped")
            continue
        doc.datums[did] = Datum(id=did, label=label, feature_nominal_ids=id_list(el, "FeatureNominalIds"))


def parse_datum_reference_frames(root: ET.Element, doc: QifDocument) -> None:
    container = find(root, "DatumReferenceFrames")
    if container is None:
        return
    for el in findall(container, "DatumReferenceFrame"):
        drf_id = el.get("id")
        if drf_id is None:
            continue
        entries: list[tuple[str, str]] = []
        datums_container = find(el, "Datums")
        if datums_container is not None:
            for datum_el in findall(datums_container, "Datum"):
                simple = find(datum_el, "SimpleDatum")
                definition_id = text_of(simple, "DatumDefinitionId") if simple is not None else None
                precedence = text_of(datum_el, "Precedence/PrecedenceEnum")
                if definition_id is not None:
                    entries.append((definition_id, precedence or ""))
        doc.datum_reference_frames[drf_id] = DatumReferenceFrame(
            id=drf_id, entries=entries, attributes=attribute_values(el),
        )


def _extract_raw_extra(el: ET.Element, skip_tags: set[str]) -> dict[str, str]:
    extra: dict[str, str] = {}
    for child in el:
        tag = local_tag(child)
        if tag in skip_tags or tag == "Id":
            continue
        if child.text and child.text.strip():
            extra[tag] = child.text.strip()
        else:
            # Wrapper element with only leaf children (e.g. Width's
            # <Tolerance><MaxValue>.2</MaxValue><MinValue>-.2</MinValue></Tolerance>)
            # -- flatten one level rather than dropping it.
            for grandchild in child:
                gtag = local_tag(grandchild)
                if grandchild.text and grandchild.text.strip():
                    extra[f"{tag}.{gtag}"] = grandchild.text.strip()
    return extra


_NOMINAL_CORE_TAGS = {"CharacteristicDefinitionId", "FeatureNominalIds", "CharacteristicDesignator", "Attributes"}


def _parse_zone_shape(el: ET.Element) -> str | None:
    # <ZoneShape> wraps an empty child naming the shape (e.g.
    # <ZoneShape><DiametricalZone/></ZoneShape>) rather than holding text
    # itself, so text_of() can't read it directly.
    zone_shape_el = find(el, "ZoneShape")
    if zone_shape_el is None:
        return None
    for child in zone_shape_el:
        tag = local_tag(child)
        if tag == "DiametricalZone":
            return "DIAMETRICAL"
        if tag == "NonDiametricalZone":
            return "NON_DIAMETRICAL"
    return None


def _parse_definition(el: ET.Element, type_name: str) -> CharacteristicDefinition:
    return CharacteristicDefinition(
        id=el.get("id", ""),
        type_name=type_name,
        tolerance_value=parse_float(el, "ToleranceValue"),
        datum_reference_frame_id=text_of(el, "DatumReferenceFrameId"),
        material_condition=text_of(el, "MaterialCondition"),
        zone_shape=_parse_zone_shape(el),
        common_zone=text_of(el, "CommonZone") == "true",
        raw_extra=_extract_raw_extra(el, _DEFINITION_CORE_TAGS),
    )


def _parse_nominal(el: ET.Element, type_name: str) -> CharacteristicNominal | None:
    definition_id = text_of(el, "CharacteristicDefinitionId")
    designator = text_of(el, "CharacteristicDesignator/Designator")
    if definition_id is None:
        return None
    return CharacteristicNominal(
        id=el.get("id", ""),
        type_name=type_name,
        definition_id=definition_id,
        feature_nominal_ids=id_list(el, "FeatureNominalIds"),
        designator=designator or "",
        raw_extra=_extract_raw_extra(el, _NOMINAL_CORE_TAGS),
    )


def _parse_item(el: ET.Element, type_name: str) -> CharacteristicItem | None:
    nominal_id = text_of(el, "CharacteristicNominalId")
    if nominal_id is None:
        return None
    return CharacteristicItem(
        id=el.get("id", ""),
        type_name=type_name,
        nominal_id=nominal_id,
        feature_item_ids=id_list(el, "FeatureItemIds"),
        measured=parse_float(el, "Measured"),
    )


def parse_characteristics(root: ET.Element, doc: QifDocument) -> None:
    characteristics = find(root, "Characteristics")
    if characteristics is None:
        return

    definitions = find(characteristics, "CharacteristicDefinitions")
    if definitions is not None:
        for el in list(definitions):
            match = _TRIAD_RE.match(local_tag(el))
            if not match:
                doc.warn(f"Unrecognized characteristic definition tag '{local_tag(el)}'; skipped")
                continue
            defn = _parse_definition(el, match.group("type_name"))
            doc.characteristic_definitions[defn.id] = defn

    nominals = find(characteristics, "CharacteristicNominals")
    if nominals is not None:
        for el in list(nominals):
            match = _TRIAD_RE.match(local_tag(el))
            if not match:
                doc.warn(f"Unrecognized characteristic nominal tag '{local_tag(el)}'; skipped")
                continue
            nom = _parse_nominal(el, match.group("type_name"))
            if nom is None:
                doc.warn(f"{local_tag(el)} {el.get('id')} missing definition reference; skipped")
                continue
            doc.characteristic_nominals[nom.id] = nom

    items = find(characteristics, "CharacteristicItems")
    if items is not None:
        for el in list(items):
            match = _TRIAD_RE.match(local_tag(el))
            if not match:
                doc.warn(f"Unrecognized characteristic item tag '{local_tag(el)}'; skipped")
                continue
            item = _parse_item(el, match.group("type_name"))
            if item is None:
                doc.warn(f"{local_tag(el)} {el.get('id')} missing nominal reference; skipped")
                continue
            doc.characteristic_items[item.id] = item
