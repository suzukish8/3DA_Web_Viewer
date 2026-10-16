"""ISO GPS data model: datums, datum reference frames, and characteristic triads.

Characteristic types are represented generically (`type_name` is the QIF tag
prefix, e.g. "Position", "Flatness", "SurfaceProfile", "Width", or any future
type such as "Circularity") so a brand-new characteristic type parses with no
code changes -- see parsing/gdt_parser.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Datum:
    id: str
    label: str
    feature_nominal_ids: list[str]


@dataclass
class DatumReferenceFrame:
    id: str
    # (datum_definition_id, precedence e.g. "PRIMARY"/"SECONDARY"/"TERTIARY")
    entries: list[tuple[str, str]]
    # Vendor-extension <AttributeStr name="..." value="..."/> children of
    # this element's own <Attributes> block (see parsing/xml_utils.py's
    # attribute_values) -- notably "_kotem.NominalMx"/"_kotem.NominalDOF",
    # the authoring software's own precomputed datum-system transform; see
    # geometry/datum_system.py.
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class CharacteristicDefinition:
    id: str
    type_name: str
    tolerance_value: Optional[float]
    datum_reference_frame_id: Optional[str]
    material_condition: Optional[str]
    # "DIAMETRICAL" (tolerance zone is a cylinder -> shown with an "⌀"
    # prefix on the value, per ISO 1101) or "NON_DIAMETRICAL", or None if
    # the type doesn't express one at all.
    zone_shape: Optional[str]
    # True if <CommonZone>true</CommonZone> -- shown as a trailing "CZ"
    # modifier in the feature control frame's tolerance compartment.
    common_zone: bool = False
    raw_extra: dict[str, str] = field(default_factory=dict)


@dataclass
class CharacteristicNominal:
    id: str
    type_name: str
    definition_id: str
    feature_nominal_ids: list[str]
    designator: str
    raw_extra: dict[str, str] = field(default_factory=dict)


@dataclass
class CharacteristicItem:
    id: str
    type_name: str
    nominal_id: str
    feature_item_ids: list[str]
    measured: Optional[float]
