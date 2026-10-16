"""Presentation-only registry for ISO GPS characteristic types, keyed on the
string `type_name` that parsing/gdt_parser.py already extracts generically.

Parsing a brand-new characteristic type needs no code changes (see
gdt_parser.py's tag-suffix regex); only *how it displays* needs registering
here. Anything not registered still renders -- as a generic gray badge, via
`get_spec`'s fallback -- so an unfamiliar type degrades gracefully instead
of vanishing or crashing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from qif_viewer.model.gdt import CharacteristicDefinition, CharacteristicItem, CharacteristicNominal

LabelFormatter = Callable[[CharacteristicNominal, CharacteristicDefinition, Optional[CharacteristicItem]], str]


@dataclass
class CharacteristicTypeSpec:
    type_name: str
    symbol_ascii: str
    symbol_unicode: Optional[str]
    color: tuple[float, float, float]
    label_formatter: LabelFormatter
    # Which icon ui/fcf_widget.py should hand-draw for this type inside its
    # ISO 1101 feature control frame box ("position", "flatness",
    # "profile_surface", ...). "generic" draws a plain "?" glyph as a safe
    # fallback for a type with no icon registered. "dimension" opts out of
    # the boxed frame entirely -- for characteristics like Width that ISO
    # 1101 doesn't express as a geometric control, just a plain toleranced
    # size -- and renders as a plain "<->" value label instead.
    symbol_kind: str = "generic"
    # Icon shown in front of the value for symbol_kind == "dimension" (a
    # plain, unboxed size/distance characteristic) -- e.g. "⌀" for a
    # diameter, "R" for a radius. Empty (the default) for a plain linear
    # width/length, which doesn't get an icon at all.
    dimension_icon: str = ""


_REGISTRY: dict[str, CharacteristicTypeSpec] = {}
_UNREGISTERED_WARNED: set[str] = set()

_FALLBACK_COLOR = (0.6, 0.6, 0.6)


def register_characteristic_type(spec: CharacteristicTypeSpec) -> None:
    _REGISTRY[spec.type_name] = spec


def _default_label(nominal: CharacteristicNominal, definition: CharacteristicDefinition, item: Optional[CharacteristicItem]) -> str:
    return f"{definition.tolerance_value:g}" if definition.tolerance_value is not None else "?"


def get_spec(type_name: str, warnings: Optional[list[str]] = None) -> CharacteristicTypeSpec:
    spec = _REGISTRY.get(type_name)
    if spec is not None:
        return spec
    if warnings is not None and type_name not in _UNREGISTERED_WARNED:
        _UNREGISTERED_WARNED.add(type_name)
        warnings.append(f"Characteristic type '{type_name}' has no presentation registered; using generic badge")
    return CharacteristicTypeSpec(
        type_name=type_name,
        symbol_ascii=f"[{type_name[:3].upper()}]",
        symbol_unicode=None,
        color=_FALLBACK_COLOR,
        label_formatter=_default_label,
    )
