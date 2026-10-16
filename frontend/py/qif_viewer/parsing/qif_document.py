"""Top-level orchestrator: XML file path -> fully populated QifDocument."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from qif_viewer.model.document import QifDocument

from .geometry_parser import parse_geometry
from .feature_parser import parse_features
from .gdt_parser import parse_characteristics, parse_datum_reference_frames, parse_datums
from .source_lines import build_line_index


def load_qif_document(path: str) -> QifDocument:
    tree = ET.parse(path)
    root = tree.getroot()

    doc = QifDocument(source_path=path)
    doc.source_lines = build_line_index(path)
    parse_datums(root, doc)
    parse_datum_reference_frames(root, doc)
    parse_features(root, doc)
    parse_characteristics(root, doc)
    parse_geometry(root, doc)
    return doc
