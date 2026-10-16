"""QifDocument: the parsed, in-memory representation of a QIF file.

Every downstream layer (tessellation, annotations, UI) consumes only this
object -- nothing downstream touches raw XML again.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .brep import Body, Curve, Edge, Face, Loop, Surface, Vertex
from .features import FeatureNominal
from .gdt import (
    CharacteristicDefinition,
    CharacteristicItem,
    CharacteristicNominal,
    Datum,
    DatumReferenceFrame,
)


@dataclass
class QifDocument:
    source_path: str = ""

    vertices: dict[str, Vertex] = field(default_factory=dict)
    curves: dict[str, Curve] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    loops: dict[str, Loop] = field(default_factory=dict)
    faces: dict[str, Face] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    bodies: dict[str, Body] = field(default_factory=dict)

    feature_nominals: dict[str, FeatureNominal] = field(default_factory=dict)

    datums: dict[str, Datum] = field(default_factory=dict)
    datum_reference_frames: dict[str, DatumReferenceFrame] = field(default_factory=dict)
    characteristic_definitions: dict[str, CharacteristicDefinition] = field(default_factory=dict)
    characteristic_nominals: dict[str, CharacteristicNominal] = field(default_factory=dict)
    characteristic_items: dict[str, CharacteristicItem] = field(default_factory=dict)

    # id -> (start_line, end_line), 1-based, from parsing/source_lines.py.
    # Lets the UI highlight "where in the raw XML" an entity came from.
    source_lines: dict[str, tuple[int, int]] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @classmethod
    def load(cls, path: str) -> "QifDocument":
        from qif_viewer.parsing.qif_document import load_qif_document
        return load_qif_document(path)
