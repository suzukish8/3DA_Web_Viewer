"""Normalise raw NX/JT PMI property bags into viewer-friendly records.

The type names intentionally match the ones ``qif_viewer.step.pmi`` writes
into ``CharacteristicDefinition.type_name`` so JT- and STEP-derived PMI can
be compared or consumed by the same code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .pmi import (
    KIND_DATUM,
    KIND_DIM,
    KIND_FCF,
    KIND_FINISH,
    KIND_NOTE,
    KIND_SYMBOL,
    KIND_TARGET,
    JtPmi,
    JtPmiEntity,
)

# ``characteristic`` enumeration from the JT V10.6 File Format Reference
# (PMIPropertiesStrings_alphabetic_list-JT_V10_6.xlsx, FCF ``characteristic``).
# Codes 2, 3, 5, 9-13 were also matched value-for-value against the STEP
# AP242 export of the same model (03knuckle / 04Housing); 1 appears in the
# JAMA samples.  Names follow step/pmi.py so JT- and STEP-derived PMI agree.
CHARACTERISTIC_NAMES = {
    0: "LineProfile",
    1: "CircularRunout",
    2: "Perpendicularity",
    3: "Position",
    4: "TotalRunout",
    5: "SurfaceProfile",
    6: "Cylindricity",
    7: "Symmetry",
    8: "Angularity",
    9: "Parallelism",
    10: "Concentricity",
    11: "Flatness",
    12: "Circularity",
    13: "Straightness",
    14: "AxisIntersection",
}

# ``ToleranceCompartment[n].zoneShape``.
ZONE_SHAPES = {0: "DIAMETRICAL", 1: "SPHERICAL_DIAMETRICAL", 2: "SQUARE"}
# ``...modifier`` (tolerance and datum references): 0 = lmc, 1 = mmc, 2 = rfs.
# NX omits the key when there is no modifier, so 0 really means LMC.
MODIFIERS = {0: "MINIMUM", 1: "MAXIMUM", 2: "REGARDLESS_OF_FEATURE_SIZE"}

# Dimension ``type`` (spec, PMI_DIM_TYPE 0x0082).  ``Ely.DimType`` is an
# NX-private code that is not in the reference and is kept apart as type_code.
DIMENSION_KINDS = {0: "CurveLength", 1: "Linear", 2: "Angular", 3: "Radial"}


@dataclass
class DatumRef:
    label: str
    modifier: Optional[str] = None


@dataclass
class JtCharacteristic:
    jt_id: int
    type_name: str
    code: int
    tolerance_value: Optional[float]
    primary: list[DatumRef] = field(default_factory=list)
    secondary: list[DatumRef] = field(default_factory=list)
    tertiary: list[DatumRef] = field(default_factory=list)
    zone_shape: Optional[str] = None
    material_condition: Optional[str] = None
    common_zone: bool = False
    unequal_value: Optional[float] = None
    text_origin: Optional[tuple[float, float, float]] = None
    # (terminator point, raw NX reference) per leader, in model coordinates.
    leaders: list[tuple[Optional[tuple[float, float, float]], str]] = field(default_factory=list)

    @property
    def datum_labels(self) -> list[str]:
        return [d.label for lvl in (self.primary, self.secondary, self.tertiary) for d in lvl]


@dataclass
class JtDimension:
    jt_id: int
    value: Optional[float]
    text: str
    upper_delta: Optional[float]
    lower_delta: Optional[float]
    type_code: Optional[int]
    kind: Optional[str] = None
    text_origin: Optional[tuple[float, float, float]] = None
    leaders: list[tuple[Optional[tuple[float, float, float]], str]] = field(default_factory=list)


@dataclass
class JtDatum:
    jt_id: int
    label: str
    is_target: bool
    target_index: Optional[int]
    text_origin: Optional[tuple[float, float, float]] = None
    leaders: list[tuple[Optional[tuple[float, float, float]], str]] = field(default_factory=list)


@dataclass
class JtNote:
    jt_id: int
    kind: str
    text: str
    text_origin: Optional[tuple[float, float, float]] = None


@dataclass
class JtSemanticPmi:
    path: str
    characteristics: list[JtCharacteristic] = field(default_factory=list)
    dimensions: list[JtDimension] = field(default_factory=list)
    datums: list[JtDatum] = field(default_factory=list)
    notes: list[JtNote] = field(default_factory=list)


def _f(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _i(v: Optional[str]) -> Optional[int]:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def _leaders(e: JtPmiEntity):
    out = []
    n = 0
    while f"Leader[{n}]" in e.props or f"Leader[{n}].terminator" in e.props or f"Leader[{n}].Reference" in e.props:
        out.append((e.vec(f"Leader[{n}].terminator"), e.props.get(f"Leader[{n}].Reference", "")))
        n += 1
    return out


def _datum_level(e: JtPmiEntity, comp: int, level: str) -> list[DatumRef]:
    refs: list[DatumRef] = []
    m = 0
    while True:
        base = f"ToleranceCompartment[{comp}].{level}Datum.Reference[{m}]"
        if base + ".label" not in e.props:
            break
        mod = _i(e.props.get(base + ".modifier"))
        refs.append(DatumRef(e.props[base + ".label"], MODIFIERS.get(mod) if mod is not None else None))
        m += 1
    # NX repeats a label as Reference[1] to carry the modifier of a datum;
    # collapse consecutive duplicates, keeping any modifier found.
    out: list[DatumRef] = []
    for r in refs:
        if out and out[-1].label == r.label:
            out[-1].modifier = out[-1].modifier or r.modifier
        else:
            out.append(r)
    return out


def _note_text(e: JtPmiEntity) -> str:
    parts = []
    n = 0
    while True:
        k = f"Text[0].Item[{n}].string"
        if k not in e.props and f"Text[0].Item[{n + 1}].string" not in e.props:
            break
        v = e.props.get(k, "")
        parts.append("\n" if v == "/n" else v)
        n += 1
    return "".join(parts)


def to_semantic(pmi: JtPmi) -> JtSemanticPmi:
    out = JtSemanticPmi(pmi.path)
    for e in pmi.entities:
        if e.kind == KIND_FCF:
            code = _i(e.props.get("characteristic")) or 0
            mod = _i(e.props.get("ToleranceCompartment[0].modifier"))
            zs = _i(e.props.get("ToleranceCompartment[0].zoneShape"))
            out.characteristics.append(JtCharacteristic(
                jt_id=e.id,
                type_name=CHARACTERISTIC_NAMES.get(code, f"Unknown({code})"),
                code=code,
                tolerance_value=_f(e.props.get("ToleranceCompartment[0].value")),
                primary=_datum_level(e, 0, "Primary"),
                secondary=_datum_level(e, 0, "Secondary"),
                tertiary=_datum_level(e, 0, "Tertiary"),
                zone_shape=ZONE_SHAPES.get(zs) if zs is not None else None,
                material_condition=MODIFIERS.get(mod) if mod is not None else None,
                common_zone=e.props.get("ToleranceCompartment[0].commonZone") == "1",
                unequal_value=_f(e.props.get("ToleranceCompartment[0].unequalValue")),
                text_origin=e.text_origin,
                leaders=_leaders(e),
            ))
        elif e.kind == KIND_DIM:
            out.dimensions.append(JtDimension(
                jt_id=e.id,
                value=_f(e.props.get("value")),
                text=e.props.get("ValueText.Item[0].string", ""),
                upper_delta=_f(e.props.get("upperDelta")),
                lower_delta=_f(e.props.get("lowerDelta")),
                type_code=_i(e.props.get("Ely.DimType") or e.props.get("type")),
                kind=DIMENSION_KINDS.get(_i(e.props.get("type"))),
                text_origin=e.text_origin,
                leaders=_leaders(e),
            ))
        elif e.kind in (KIND_DATUM, KIND_TARGET):
            out.datums.append(JtDatum(
                jt_id=e.id,
                label=e.props.get("label", ""),
                is_target=e.kind == KIND_TARGET,
                target_index=_i(e.props.get("index")),
                text_origin=e.text_origin,
                leaders=_leaders(e),
            ))
        elif e.kind in (KIND_NOTE, KIND_FINISH, KIND_SYMBOL):
            out.notes.append(JtNote(e.id, e.kind, _note_text(e), e.text_origin))
    return out
