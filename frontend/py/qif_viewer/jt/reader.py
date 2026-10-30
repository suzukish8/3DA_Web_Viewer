"""Minimal JT container reader: file header, table of contents (TOC) and
decompressed segment payloads.

Only what PMI extraction needs. Layout (JT 9.x/10.x, little-endian):
  80-byte version string, byte-order U8, reserved I32,
  TOC offset (I32 for <10, I64 for >=10), LSG segment GUID (16).
  TOC: count I32, then per entry: GUID(16), offset (I32/I64), length I32,
  attributes U32 (segment type = attributes >> 24).
  Segment: GUID(16) type I32 length I32, then payload whose first bytes are
  compression-flag I32 (1 none / 2 zlib / 3 lzma), compressed-length I32
  (includes the 1-byte algorithm id), algorithm U8.
"""
from __future__ import annotations

import lzma
import struct
import zlib
from dataclasses import dataclass, field

SEG_LSG = 1
SEG_PMI = 3
SEG_META = 4
SEG_XT_BREP = 17

# Segment-type id -> readable name (the subset seen in practice).
SEGMENT_NAMES = {
    1: "LSG", 2: "JT B-Rep", 3: "PMI Data", 4: "Meta Data", 6: "Shape",
    7: "Shape LOD0", 8: "Shape LOD1", 9: "Shape LOD2", 10: "Shape LOD3",
    17: "XT B-Rep", 18: "Wireframe",
}


@dataclass
class JtSegment:
    guid: bytes
    offset: int
    length: int
    type: int


@dataclass
class JtFile:
    path: str
    version: str
    byte_order: str  # struct prefix, '<' or '>'
    data: bytes = field(repr=False)
    segments: list[JtSegment] = field(default_factory=list)

    def segments_of(self, seg_type: int) -> list[JtSegment]:
        return [s for s in self.segments if s.type == seg_type]

    def payload(self, seg: JtSegment) -> bytes:
        """Decompressed element stream of one segment."""
        e = self.byte_order
        raw = self.data[seg.offset:seg.offset + seg.length]
        body_len = struct.unpack_from(e + "i", raw, 20)[0]
        body = raw[24:24 + body_len]
        flag, clen = struct.unpack_from(e + "ii", body, 0)
        comp = body[9:9 + clen - 1]  # byte 8 is the algorithm id
        if flag == 3:
            return lzma.LZMADecompressor().decompress(comp)
        if flag == 2:
            return zlib.decompress(comp)
        return comp


def read_jt(path: str) -> JtFile:
    with open(path, "rb") as fh:
        d = fh.read()
    version = d[:80].decode("ascii", "replace").strip()
    if not version.startswith("Version"):
        raise ValueError(f"not a JT file: {path}")
    e = "<" if d[80] == 0 else ">"
    major = int(version.split()[1].split(".")[0])
    p = 85
    if major >= 10:
        toc_off = struct.unpack_from(e + "q", d, p)[0]
    else:
        toc_off = struct.unpack_from(e + "i", d, p)[0]
    n = struct.unpack_from(e + "i", d, toc_off)[0]
    q = toc_off + 4
    jt = JtFile(path, version, e, d)
    for _ in range(n):
        guid = d[q:q + 16]
        q += 16
        if major >= 10:
            off = struct.unpack_from(e + "q", d, q)[0]
            q += 8
        else:
            off = struct.unpack_from(e + "i", d, q)[0]
            q += 4
        length, attr = struct.unpack_from(e + "iI", d, q)
        q += 8
        jt.segments.append(JtSegment(guid, off, length, attr >> 24))
    return jt
