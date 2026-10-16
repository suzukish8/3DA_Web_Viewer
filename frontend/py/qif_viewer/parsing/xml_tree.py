"""Builds a full element tree of a QIF/XML source string, each node keeping
its start/end line so the UI can jump from a tree node straight to "where in
the raw text this came from" -- the same idea as source_lines.py's id-keyed
index, just organized as a tree (parent/children) instead of a flat dict, and
keyed by position in the document rather than by `id` attribute (so it also
covers elements that have no `id`, e.g. <QIFDocument>, <Header>, ...).

Used by ui/structure_tree.py to populate the Debug > "Show QIF
Structure Tree" panel -- an Explorer-style view of the loaded file's element
hierarchy -- and by ui/main_window.py to keep that panel, the QIF Source
panel, the ISO GPS sidebar, the 3D view, and annotation clicks all cross-linked
(see its _reveal_related_selection/_reveal_structure_node_for_line/_for_id).

Same expat-over-ElementTree tradeoff as source_lines.py: the C-accelerated
xml.etree.ElementTree used everywhere else in this codebase doesn't expose
line numbers to Python, so this does its own separate pass with
xml.parsers.expat, which does.
"""
from __future__ import annotations

import re
import xml.parsers.expat as expat
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class XmlTreeNode:
    tag: str
    attrib: dict[str, str]
    start_line: int
    end_line: int
    children: list["XmlTreeNode"] = field(default_factory=list)
    # This element's own character data, e.g. the "12.5" in <Value>12.5</Value>
    # -- only meaningful for a leaf (no children): for an element that itself
    # has child elements, expat still reports the whitespace between those
    # children's tags as character data attributed to this element, so
    # `text` for a non-leaf node is just pretty-printing indentation, not
    # real content. ui/structure_tree.py's detail panel only ever reads this
    # for childless nodes, for exactly that reason.
    text: str = ""
    # Back-reference to the enclosing element (None for the document root)
    # -- lets QifStructureTreeWidget.reveal_node walk from an arbitrary node
    # back up to the root to find the chain of ancestors it needs to lazily
    # expand. Excluded from dataclass-generated __repr__/__eq__: comparing
    # (or printing) `parent` would walk back into this same node via the
    # parent's own `children` list, so with the default compare=True/
    # repr=True this would recurse forever the moment a node has both a
    # parent and children (i.e. every non-root, non-leaf node).
    parent: Optional["XmlTreeNode"] = field(default=None, repr=False, compare=False)


def build_xml_tree(text: str) -> tuple[XmlTreeNode, dict[str, "XmlTreeNode"]]:
    """Parses `text` (a full QIF/XML document) into an XmlTreeNode tree
    rooted at its document element, alongside an {id: XmlTreeNode} index of
    every element that carries an `id` attribute (the same key space as
    source_lines.py's own index, built here in the same single pass instead
    of a second one). Raises xml.parsers.expat.ExpatError if `text` isn't
    well-formed XML -- callers should catch that the same way QifTextView.
    _validate_syntax already does for the same string."""
    root: XmlTreeNode | None = None
    stack: list[XmlTreeNode] = []
    id_index: dict[str, XmlTreeNode] = {}

    parser = expat.ParserCreate()

    def start(name: str, attrs: dict) -> None:
        node = XmlTreeNode(
            tag=name,
            attrib=dict(attrs),
            start_line=parser.CurrentLineNumber,
            end_line=parser.CurrentLineNumber,
            parent=stack[-1] if stack else None,
        )
        if stack:
            stack[-1].children.append(node)
        stack.append(node)
        id_value = attrs.get("id")
        if id_value is not None:
            id_index[id_value] = node

    def end(name: str) -> None:  # noqa: ARG001 - expat's callback signature
        nonlocal root
        node = stack.pop()
        node.end_line = parser.CurrentLineNumber
        if not stack:
            root = node

    def chardata(data: str) -> None:
        if stack:
            stack[-1].text += data

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = chardata
    parser.Parse(text, True)

    if root is None:
        # Well-formed-but-empty input (or expat accepted something with no
        # element at all) -- treat the same as a parse error rather than
        # returning None, so callers only have one failure path to handle.
        raise expat.ExpatError("no root element found")
    return root, id_index


def find_node_at_line(root: XmlTreeNode, line: int) -> Optional[XmlTreeNode]:
    """The deepest element in `root`'s tree whose [start_line, end_line]
    contains `line`, or None if `line` falls outside root's own range
    entirely. Used to resolve "the cursor is on line N of the QIF Source
    panel" (or "this Annotation's defining element starts at line N") back
    to a tree node -- the reverse of clicking a node to jump to its lines."""
    if not (root.start_line <= line <= root.end_line):
        return None
    for child in root.children:
        found = find_node_at_line(child, line)
        if found is not None:
            return found
    return root


_STEP_ENTITY_START_RE = re.compile(r"\s*(?:#(\d+)\s*=\s*)?(\(|[A-Za-z_][A-Za-z0-9_]*)")


def build_step_tree(text: str) -> tuple[XmlTreeNode, dict[str, "XmlTreeNode"]]:
    """STEP (ISO 10303-21) counterpart of build_xml_tree: ISO-10303-21 root
    -> HEADER / DATA sections -> one node per entity instance, in file order
    (so find_node_at_line's line lookup works unchanged). Data entities carry
    id="#123" (also keyed in the returned index) and their full statement as
    `text`, so the structure tree's content panel shows the entity itself.
    A plain line scan -- a ';' ending a line inside a string literal would
    split an entity early, which only affects its end line."""
    lines = text.splitlines()
    root = XmlTreeNode(tag="ISO-10303-21", attrib={}, start_line=1, end_line=max(len(lines), 1))
    id_index: dict[str, XmlTreeNode] = {}
    section: XmlTreeNode | None = None
    entity_start = 0
    entity_lines: list[str] = []
    in_comment = False

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if entity_lines:
            entity_lines.append(raw)
        elif in_comment:
            if "*/" in stripped:
                in_comment = False
            continue
        else:
            if not stripped:
                continue
            if stripped.startswith("/*") and "*/" not in stripped:
                in_comment = True
                continue
            upper = stripped.upper()
            if upper in ("HEADER;", "DATA;") or upper.startswith("DATA;"):
                section = XmlTreeNode(
                    tag=upper.rstrip(";"), attrib={}, start_line=lineno, end_line=lineno, parent=root)
                root.children.append(section)
                continue
            if upper.startswith("ENDSEC"):
                if section is not None:
                    section.end_line = lineno
                    section = None
                continue
            if section is None or not _STEP_ENTITY_START_RE.match(stripped):
                continue
            entity_start = lineno
            entity_lines = [raw]
        if entity_lines and stripped.endswith(";"):
            statement = "\n".join(entity_lines)
            m = _STEP_ENTITY_START_RE.match(statement)
            name = "COMPLEX_ENTITY" if m.group(2) == "(" else m.group(2)
            attrib = {"id": f"#{m.group(1)}"} if m.group(1) else {}
            node = XmlTreeNode(
                tag=name, attrib=attrib, start_line=entity_start, end_line=lineno,
                text=statement.strip(), parent=section)
            section.children.append(node)
            if attrib:
                id_index[attrib["id"]] = node
            entity_lines = []
    return root, id_index
