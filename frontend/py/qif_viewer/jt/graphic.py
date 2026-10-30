"""Graphic PMI from JT "PMI Data": the drawn lines of every Generic PMI Entity
(feature-control frames, dimension / leader lines, arrows ...).

Each Generic PMI Entity (spec 8.3.6) is stored as

    U32 type-name string id, U32 parent-type string id,
    U16 entity type, U16 parent type, U16 user flags,
    PMI 2D Data:  I32 user label, U8 frame flag, [3 x CoordF32 frame],
                  F32 text height, U8 valid flag,
                  I32 text entity count, text entities...,
                  Non-Text Polyline Data (index / type / width arrays and the
                  packed XYZ vertex list),
    I32 property count, then the key/value property pairs that
    ``jt.pmi`` already reads.

The entity list is contiguous, so once one entity start is known the rest
are chained forward and every step is validated against the property count.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .pmi import _MARK_CNT, _MARK_TOL, _pair
from .reader import SEG_PMI, JtFile

POLY_TYPE_NAMES = {
    0: "line", 1: "arrow", 2: "circle", 3: "arc", 4: "ext_line1", 5: "ext_line2",
    6: "ext_arc", 7: "ext_circle", 8: "text_line", 9: "text_string",
}


@dataclass
class GraphicPolyline:
    type: int
    width: int
    points: np.ndarray            # (N, 3)


@dataclass
class GraphicEntity:
    run_start: int                # offset of the entity's first property pair in the segment payload
    entity_type: int
    label: int
    text_height: float
    frame: tuple                  # (origin, x point, y point) or None
    polylines: list[GraphicPolyline] = field(default_factory=list)


def _runs(b: bytes):
    """(start, end, pair_count) of every property run whose header count matches."""
    p, n = 0, len(b)
    while p < n:
        if not _pair(b, p):
            p += 1
            continue
        start = p
        cnt = 0
        while True:
            r = _pair(b, p)
            if not r:
                break
            p = r[2]
            cnt += 1
        if start >= 4 and struct.unpack_from("<i", b, start - 4)[0] == cnt:
            yield start, p, cnt


class _Fail(Exception):
    pass


# Garbage offsets raise different errors on different platforms (the 32-bit
# Pyodide build raises OverflowError where CPython raises struct.error).
_PARSE_ERRORS = (_Fail, struct.error, OverflowError, ValueError, IndexError)


def _parse_entity(b: bytes, p: int):
    """Parse one Generic PMI Entity from its header at ``p``.
    Returns (GraphicEntity-without-run_start, properties start, properties end)."""
    n = len(b)

    def need(k):
        if p + k > n:
            raise _Fail

    need(14 + 4 + 1)
    etype, _parent, _flags = struct.unpack_from("<HHH", b, p + 8)
    q = p + 14
    label = struct.unpack_from("<i", b, q)[0]
    flag = b[q + 4]
    q += 5
    frame = None
    if flag:
        if q + 36 > n:
            raise _Fail
        v = struct.unpack_from("<9f", b, q)
        frame = (v[0:3], v[3:6], v[6:9])
        q += 36
    height = struct.unpack_from("<f", b, q)[0]
    q += 5                                   # height + valid flag
    ntext = struct.unpack_from("<i", b, q)[0]
    q += 4
    if ntext < 0 or ntext > 200:
        raise _Fail
    for _ in range(ntext):                   # 2D Text Data (glyph placement lives elsewhere)
        q += 4 + 4 + 4 + 4 + 24
        if q + 4 > n:
            raise _Fail
        k = struct.unpack_from("<i", b, q)[0]
        q += 4
        if k < 0 or k > 100000:
            raise _Fail
        if k > 0:
            q += 2 * k
            if q + 4 > n:
                raise _Fail
            nv = struct.unpack_from("<i", b, q)[0]
            if nv < 0 or q + 4 + 4 * nv > n:
                raise _Fail
            q += 4 + 4 * nv
    if q + 4 > n:
        raise _Fail
    ni = struct.unpack_from("<i", b, q)[0]
    if ni < 0 or ni > 100000 or q + 4 + 4 * ni > n:
        raise _Fail
    idx = struct.unpack_from(f"<{ni}i", b, q + 4)
    q += 4 + 4 * ni
    nt = struct.unpack_from("<i", b, q)[0]
    if nt < 0 or nt > 100000:
        raise _Fail
    types = struct.unpack_from(f"<{nt}h", b, q + 4)
    q += 4 + 2 * nt
    nw = struct.unpack_from("<i", b, q)[0]
    if nw < 0 or nw > 100000:
        raise _Fail
    widths = struct.unpack_from(f"<{nw}h", b, q + 4)
    q += 4 + 2 * nw
    nc = struct.unpack_from("<i", b, q)[0]
    if nc < 0 or nc % 3 or q + 4 + 4 * nc > n:
        raise _Fail
    coords = np.frombuffer(b, dtype="<f4", count=nc, offset=q + 4).reshape(-1, 3).astype(np.float64)
    q += 4 + 4 * nc
    props_count = struct.unpack_from("<i", b, q)[0]
    props_start = q + 4

    polylines: list[GraphicPolyline] = []
    for i in range(len(idx) - 1):
        a, c = idx[i], idx[i + 1]
        if not (0 <= a < c <= len(coords)):
            continue
        ty = next((t for lim, t in zip(types[0::2], types[1::2]) if i < lim), 0)
        wd = next((w for lim, w in zip(widths[0::2], widths[1::2]) if i < lim), 1)
        polylines.append(GraphicPolyline(ty, wd, coords[a:c]))
    ent = GraphicEntity(0, etype, label, height, frame, polylines)
    return ent, props_start, props_count


def _props_end(b: bytes, p: int, count: int):
    for _ in range(count):
        r = _pair(b, p)
        if not r:
            return None
        p = r[2]
    return p


def extract_graphic(jt: JtFile) -> dict[int, GraphicEntity]:
    """``{run_start: GraphicEntity}`` for the PMI-manager segment(s), keyed by
    the same property-run offsets ``jt.pmi`` records (``run_start``)."""
    out: dict[int, GraphicEntity] = {}
    for seg in jt.segments_of(SEG_PMI):
        try:
            blob = jt.payload(seg)
        except Exception:  # noqa: BLE001
            continue
        if _MARK_TOL not in blob and _MARK_CNT not in blob:
            continue
        run_starts = {s: (e, c) for s, e, c in _runs(blob)}
        ordered = sorted(run_starts)
        done: set[int] = set()
        for first in ordered:
            if first in done:
                continue
            # Entity start of ``first``: walk its 2D data backwards is
            # unreliable, so try each earlier run end as a chain anchor.
            start = _find_start(blob, first, run_starts)
            if start is None:
                continue
            p = start
            while p is not None:
                try:
                    ent, props_start, cnt = _parse_entity(blob, p)
                except _PARSE_ERRORS:
                    break
                end = _props_end(blob, props_start, cnt)
                if end is None or props_start not in run_starts:
                    break
                ent.run_start = props_start
                out[props_start] = ent
                done.add(props_start)
                p = end
    return out


def _find_start(blob: bytes, run_start: int, run_starts: dict) -> int | None:
    """Entity start for the run at ``run_start``: the end of the previous
    run when that parses, else a backward search over plausible header offsets."""
    prev_ends = [e for s, (e, _c) in run_starts.items() if e <= run_start - 4]
    if prev_ends:
        p = max(prev_ends)
        try:
            ent, props_start, _ = _parse_entity(blob, p)
            if props_start == run_start:
                return p
        except _PARSE_ERRORS:
            pass
    for back in range(40, 40000):            # fall back: scan candidate header offsets
        p = run_start - 4 - back
        if p < 0:
            break
        try:
            ent, props_start, _ = _parse_entity(blob, p)
        except _PARSE_ERRORS:
            continue
        if props_start == run_start and ent.polylines:
            return p
    return None
