"""Presentation registrations for the characteristic types found in the
sample QIF files this app was built against. Adding another type (e.g.
Circularity, Perpendicularity) is one more `register_characteristic_type`
call -- nothing else changes.
"""
from __future__ import annotations

from typing import Optional

from qif_viewer.model.gdt import CharacteristicDefinition, CharacteristicItem, CharacteristicNominal

from .registry import CharacteristicTypeSpec, register_characteristic_type


def tolerance_value_text(definition: CharacteristicDefinition) -> str:
    if definition.tolerance_value is not None:
        return f"{definition.tolerance_value:g}"
    # Not every type expresses tolerance as a single symmetric value (e.g.
    # Width's <Tolerance><MaxValue>/<MinValue>) -- fall back to whatever the
    # generic raw_extra capture (parsing/gdt_parser.py) found for it.
    max_v, min_v = definition.raw_extra.get("Tolerance.MaxValue"), definition.raw_extra.get("Tolerance.MinValue")
    if max_v is not None and min_v is not None:
        # An equal +/- tolerance ("+0.2/-0.2") is conventionally written
        # as a single "±0.2" instead -- but only when it's actually
        # symmetric; an unequal one ("+0.2/-0.1") is left as-is since
        # there's no single number that means the same thing.
        try:
            if abs(float(max_v)) == abs(float(min_v)):
                return f"±{max_v}"
        except ValueError:
            pass
        return f"+{max_v}/{min_v}"
    if "NonTolerance" in definition.raw_extra:
        return definition.raw_extra["NonTolerance"]
    return "?"


def _make_formatter():
    def formatter(nominal: CharacteristicNominal, definition: CharacteristicDefinition, item: Optional[CharacteristicItem]) -> str:
        # "MEASURED" (QIF's <NonTolerance> value for a reference dimension
        # with no tolerance at all) isn't itself a value worth displaying --
        # the actual number the user cares about is the nominal/target
        # value next to it (e.g. "1" in a Radius's "R1"), so it's dropped
        # rather than shown as literal text.
        text = tolerance_value_text(definition)
        if text == "MEASURED":
            text = ""
        target = nominal.raw_extra.get("TargetValue")
        if target is not None:
            text = f"{target} {text}".strip()
        if item is not None and item.measured is not None:
            text += f" (measured {item.measured:g})"
        return text
    return formatter


register_characteristic_type(CharacteristicTypeSpec(
    type_name="Position", symbol_ascii="[POS]", symbol_unicode="⌖",
    color=(0.90, 0.25, 0.25), label_formatter=_make_formatter(), symbol_kind="position",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Flatness", symbol_ascii="[FLT]", symbol_unicode="▱",
    color=(0.25, 0.55, 0.90), label_formatter=_make_formatter(), symbol_kind="flatness",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="SurfaceProfile", symbol_ascii="[PRF]", symbol_unicode=None,
    color=(0.30, 0.75, 0.35), label_formatter=_make_formatter(), symbol_kind="profile_surface",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Width", symbol_ascii="[WID]", symbol_unicode=None,
    color=(0.85, 0.55, 0.15), label_formatter=_make_formatter(), symbol_kind="dimension",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Perpendicularity", symbol_ascii="[PER]", symbol_unicode="⊥",
    color=(0.55, 0.35, 0.85), label_formatter=_make_formatter(), symbol_kind="perpendicularity",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Diameter", symbol_ascii="[DIA]", symbol_unicode="⌀",
    color=(0.85, 0.55, 0.15), label_formatter=_make_formatter(), symbol_kind="dimension", dimension_icon="⌀",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Radius", symbol_ascii="[RAD]", symbol_unicode=None,
    color=(0.85, 0.55, 0.15), label_formatter=_make_formatter(), symbol_kind="dimension", dimension_icon="R",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Length", symbol_ascii="[LEN]", symbol_unicode=None,
    color=(0.85, 0.55, 0.15), label_formatter=_make_formatter(), symbol_kind="dimension",
))

# Additional ISO 1101 types seen in STEP AP242 PMI (qif_viewer/step/pmi.py)
# but not any sample QIF file this app was originally built against. Their
# boxed ISO symbols are hand-drawn in ui/fcf_widget.py (Qt sidebar) and
# ui/vector_notation.py (3D view / DXF / SVG / PDF export).
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Parallelism", symbol_ascii="[PAR]", symbol_unicode="∥",
    color=(0.55, 0.35, 0.85), label_formatter=_make_formatter(), symbol_kind="parallelism",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Angularity", symbol_ascii="[ANG]", symbol_unicode="∠",
    color=(0.55, 0.35, 0.85), label_formatter=_make_formatter(), symbol_kind="angularity",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Concentricity", symbol_ascii="[CNC]", symbol_unicode="◎",
    color=(0.90, 0.25, 0.25), label_formatter=_make_formatter(), symbol_kind="concentricity",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Circularity", symbol_ascii="[CIR]", symbol_unicode="○",
    color=(0.25, 0.55, 0.90), label_formatter=_make_formatter(), symbol_kind="circularity",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Cylindricity", symbol_ascii="[CYL]", symbol_unicode=None,
    color=(0.25, 0.55, 0.90), label_formatter=_make_formatter(), symbol_kind="cylindricity",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Straightness", symbol_ascii="[STR]", symbol_unicode="⏤",
    color=(0.25, 0.55, 0.90), label_formatter=_make_formatter(), symbol_kind="straightness",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="Symmetry", symbol_ascii="[SYM]", symbol_unicode=None,
    color=(0.90, 0.25, 0.25), label_formatter=_make_formatter(), symbol_kind="symmetry",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="LineProfile", symbol_ascii="[PRL]", symbol_unicode=None,
    color=(0.30, 0.75, 0.35), label_formatter=_make_formatter(), symbol_kind="profile_line",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="CircularRunout", symbol_ascii="[CRO]", symbol_unicode=None,
    color=(0.90, 0.25, 0.25), label_formatter=_make_formatter(), symbol_kind="circular_runout",
))
register_characteristic_type(CharacteristicTypeSpec(
    type_name="TotalRunout", symbol_ascii="[TRO]", symbol_unicode=None,
    color=(0.90, 0.25, 0.25), label_formatter=_make_formatter(), symbol_kind="total_runout",
))
