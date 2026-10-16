"""Maps each QIF element's `id` attribute to its (start_line, end_line) in
the original file text, so the UI can highlight "where in the XML file"
a parsed entity (a Datum, a CharacteristicNominal, ...) came from.

xml.etree.ElementTree's C accelerator (the default in modern CPython) does
not expose source position info to Python subclasses -- the classic
"override XMLParser._start/_end" recipe only works against the pure-Python
fallback, which is not what's active here. Doing a second, separate pass
over the file with the stdlib's `xml.parsers.expat` directly (which does
report `CurrentLineNumber`) is simple and reliable, and works because every
QIF element that we care about referencing already carries a globally
unique `id` attribute -- we key the index on that.
"""
from __future__ import annotations

import xml.parsers.expat as expat


def build_line_index(path: str) -> dict[str, tuple[int, int]]:
    """Returns {id_attr_value: (start_line, end_line)}, both 1-based and
    inclusive of the element's opening/closing tags. Elements without an
    `id` attribute are not indexed (nothing in the UI needs to look them
    up by id)."""
    index: dict[str, tuple[int, int]] = {}
    stack: list[tuple[str | None, int]] = []

    parser = expat.ParserCreate()

    def start(name: str, attrs: dict) -> None:
        stack.append((attrs.get("id"), parser.CurrentLineNumber))

    def end(name: str) -> None:
        id_value, start_line = stack.pop()
        if id_value is not None:
            index[id_value] = (start_line, parser.CurrentLineNumber)

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    with open(path, "rb") as f:
        parser.ParseFile(f)
    return index
