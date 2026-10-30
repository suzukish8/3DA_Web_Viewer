"""JT Logical Scene Graph (LSG) reader: walks the node tree and returns the
tri-strip shapes with the shape-LOD segment each one loads late and its
accumulated model transform (spec 5.2-5.6).

Only what the geometry import needs: group-like nodes, instance nodes,
LOD nodes (first = most detailed child is followed), geometric-transform
attributes and the property table that links shape nodes to their Shape LOD
segments.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .reader import JtFile

GUID_END = 0xFFFFFFFF
GUID_GROUP = 0x10DD101B
GUID_INSTANCE = 0x10DD102A
GUID_LOD = 0x10DD102C
GUID_RANGE_LOD = 0x10DD104C
GUID_SWITCH = 0x10DD10F3
GUID_META = 0xCE357245
GUID_PART = 0xCE357244
GUID_PARTITION = 0x10DD103E
GUID_TRISTRIP = 0x10DD1077
GUID_TRANSFORM = 0x10DD1083
GUID_STRING_PROP = 0x10DD106E
GUID_LATE_LOADED = 0xE0B05BE5

_GROUP_LIKE = {GUID_GROUP, GUID_LOD, GUID_RANGE_LOD, GUID_META, GUID_PART, GUID_PARTITION, GUID_SWITCH}


@dataclass
class _El:
    guid: int
    base: int
    oid: int
    data: bytes  # bytes after the object id


@dataclass
class ShapeRef:
    node_id: int
    segment_guid: bytes
    segment_type: int
    matrix: np.ndarray            # 4x4, row-vector convention (p' = p @ M)
    name: str = ""


@dataclass
class LsgResult:
    shapes: list[ShapeRef] = field(default_factory=list)
    names: dict[int, str] = field(default_factory=dict)


def _read_elements(d: bytes, p: int) -> tuple[list[_El], int]:
    out: list[_El] = []
    while p + 4 <= len(d):
        length = struct.unpack_from("<i", d, p)[0]
        if length <= 0:
            break
        guid = struct.unpack_from("<I", d, p + 4)[0]
        if guid == GUID_END and d[p + 4:p + 20] == b"\xff" * 16:
            return out, p + 4 + length
        base = d[p + 20]
        oid = struct.unpack_from("<i", d, p + 21)[0]
        out.append(_El(guid, base, oid, d[p + 25:p + 4 + length]))
        p += 4 + length
    return out, p


def _node_header(data: bytes) -> tuple[int, list[int]]:
    """Base Node Data: ver U8, flags U32, attr count I32, attr ids.  Returns (end, attr ids)."""
    n = struct.unpack_from("<i", data, 5)[0]
    ids = list(struct.unpack_from(f"<{n}i", data, 9)) if n > 0 else []
    return 9 + 4 * max(n, 0), ids


def _matrix(data: bytes) -> np.ndarray:
    # Base attribute data: ver U8 + 9 bytes; transform ver I8, mask U16, F64s.
    q = 10
    q += 1
    mask = struct.unpack_from("<H", data, q)[0]
    q += 2
    m = np.eye(4)
    for bit in range(16):
        if mask & (0x8000 >> bit):
            m[bit // 4, bit % 4] = struct.unpack_from("<d", data, q)[0]
            q += 8
    return m


def read_lsg(jt: JtFile) -> LsgResult:
    segs = jt.segments_of(1)
    if not segs:
        return LsgResult()
    d = jt.payload(segs[0])
    nodes, p = _read_elements(d, 0)
    atoms, p = _read_elements(d, p)

    by_id: dict[int, _El] = {e.oid: e for e in nodes}
    atom_by_id = {e.oid: e for e in atoms}

    # ---- property table: node id -> [(key atom id, value atom id)]
    props: dict[int, list[tuple[int, int]]] = {}
    if p + 6 <= len(d):
        count = struct.unpack_from("<i", d, p + 2)[0]
        q = p + 6
        for _ in range(count):
            if q + 4 > len(d):
                break
            node = struct.unpack_from("<i", d, q)[0]
            q += 4
            lst = []
            while q + 4 <= len(d):
                key = struct.unpack_from("<i", d, q)[0]
                q += 4
                if key == 0:
                    break
                val = struct.unpack_from("<i", d, q)[0]
                q += 4
                lst.append((key, val))
            props[node] = lst

    def string_of(atom_id: int) -> str:
        a = atom_by_id.get(atom_id)
        if a is None or a.guid != GUID_STRING_PROP:
            return ""
        n = struct.unpack_from("<i", a.data, 6)[0]
        return a.data[10:10 + 2 * n].decode("utf-16-le", "replace")

    def late_loaded(atom_id: int):
        a = atom_by_id.get(atom_id)
        if a is None or a.guid != GUID_LATE_LOADED:
            return None
        guid = a.data[6:22]
        seg_type = struct.unpack_from("<i", a.data, 22)[0]
        return guid, seg_type

    def children_of(e: _El) -> list[int]:
        end, _ = _node_header(e.data)
        if e.guid == GUID_INSTANCE:
            return [struct.unpack_from("<i", e.data, end + 1)[0]]
        n = struct.unpack_from("<i", e.data, end + 1)[0]
        return list(struct.unpack_from(f"<{n}i", e.data, end + 5)) if n > 0 else []

    result = LsgResult()

    def visit(oid: int, parent: np.ndarray, name: str, depth: int) -> None:
        e = by_id.get(oid)
        if e is None or depth > 64:
            return
        _, attrs = _node_header(e.data)
        m = parent
        for aid in attrs:
            a = by_id.get(aid)
            if a is not None and a.guid == GUID_TRANSFORM:
                m = _matrix(a.data) @ m
        for key, val in props.get(oid, []):
            if string_of(key) in ("JT_PROP_NAME",):
                nm = string_of(val)
                if nm:
                    name = nm.split(";")[0]
        if e.guid == GUID_TRISTRIP:
            for key, val in props.get(oid, []):
                ll = late_loaded(val)
                if ll is not None and 6 <= ll[1] <= 16:
                    result.shapes.append(ShapeRef(oid, ll[0], ll[1], m, name))
            return
        if e.guid in _GROUP_LIKE or e.guid == GUID_INSTANCE:
            kids = children_of(e)
            if e.guid in (GUID_LOD, GUID_RANGE_LOD):
                kids = kids[:1]  # first child is the most detailed representation
            elif e.guid == GUID_SWITCH:
                kids = kids[:1]
            for k in kids:
                visit(k, m, name, depth + 1)

    root = nodes[0].oid if nodes else 0
    visit(root, np.eye(4), "", 0)
    return result
