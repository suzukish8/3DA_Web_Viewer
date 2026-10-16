"""Small helpers for reading QIF3 XML with its default namespace.

Kept intentionally dumb (plain ElementTree, no lxml dependency) -- QIF files
are a few hundred KB to a few MB, well within ElementTree's comfort zone.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

QIF_NS = "http://qifstandards.org/xsd/qif3"


def qn(tag: str) -> str:
    return f"{{{QIF_NS}}}{tag}"

def _qpath(path: str) -> str:
    return "/".join(qn(p) for p in path.split("/"))


def find(el: ET.Element, path: str) -> Optional[ET.Element]:
    return el.find(_qpath(path))


def findall(el: ET.Element, path: str) -> list[ET.Element]:
    return el.findall(_qpath(path))


def find_any(el: ET.Element, path: str) -> Optional[ET.Element]:
    """Search anywhere below el (`.//`), for elements not directly nested."""
    return el.find(".//" + _qpath(path))


def text_of(el: ET.Element, path: str) -> Optional[str]:
    found = find(el, path)
    if found is None or found.text is None:
        return None
    stripped = found.text.strip()
    return stripped or None


def local_tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()], dtype=float)


def parse_point(el: ET.Element, path: str) -> Optional[np.ndarray]:
    t = text_of(el, path)
    return parse_vec(t) if t is not None else None


def parse_float(el: ET.Element, path: str) -> Optional[float]:
    t = text_of(el, path)
    return float(t) if t is not None else None


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split())


def id_ref(el: ET.Element, path: str) -> Optional[str]:
    """A single <Path>...<Id>value</Id></Path>-shaped reference."""
    return text_of(el, path + "/Id") if find(el, path) is not None else text_of(el, path)


def ids_in(el: ET.Element) -> list[str]:
    """All direct <Id> children of el, e.g. inside a `FooIds` container."""
    return [c.text.strip() for c in el.findall(qn("Id")) if c.text]


def id_list(el: ET.Element, path: str) -> list[str]:
    container = find(el, path)
    if container is None:
        return []
    return ids_in(container)


def attribute_values(el: ET.Element, path: str = "Attributes") -> dict[str, str]:
    """{name: value} for every name/value-carrying child (AttributeStr,
    AttributeReal, ...) of el's own <path> container -- QIF's vendor-
    extension attribute pattern (e.g. "_kotem.NominalMx"), generic over the
    attribute element's own tag since only its name/value XML attributes
    matter here."""
    container = find(el, path)
    if container is None:
        return {}
    values: dict[str, str] = {}
    for child in container:
        name, value = child.get("name"), child.get("value")
        if name is not None and value is not None:
            values[name] = value
    return values
