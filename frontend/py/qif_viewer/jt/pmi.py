"""Semantic PMI extraction from JT "PMI Data" segments.

NX (JT translator) stores each PMI entity as a flat key/value property list
inside the PMI-manager element (GUID CE357249-38FB-11D1-A506-006097BDC6E1):

    [entity id I32][pair count I32] then count x
        key   MbString  (I32 char count + UTF-16LE)
        flag  U8
        value MbString
        flag  U8

Values are always strings.  Keys of interest:
  ToleranceCompartment[n].value / .PrimaryDatum.Reference[m].label / ...
  characteristic, zoneShape, commonZone, unequal*, ValueText.*, upperDelta,
  lowerDelta, label, Leader[0].terminator / .Reference, textOrigin, ...
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Optional

from .reader import JtFile, SEG_PMI

_KEY_RE = re.compile(r"^[A-Za-z_$][\w\[\]\.:/ #<>=\-]*$")
_NAME_RE = re.compile(r"^(.*?)\s*\((\d+)\)\s*$")

KIND_FCF = "FeatureControlFrame"
KIND_DIM = "Dimension"
KIND_DATUM = "DatumFeatureSymbol"
KIND_TARGET = "DatumTarget"
KIND_NOTE = "Note"
KIND_FINISH = "SurfaceFinish"
KIND_SYMBOL = "UserDefinedSymbol"
KIND_OTHER = "Other"


@dataclass
class JtPmiEntity:
    id: int
    kind: str
    name: str                      # e.g. "Feature Control Frame (35)" (may be "")
    props: dict[str, str] = field(default_factory=dict)
    segment_offset: int = 0
    run_start: int = 0             # offset of the property run in the PMI payload (joins to jt.graphic)

    def vec(self, key: str) -> Optional[tuple[float, float, float]]:
        v = self.props.get(key)
        if not v:
            return None
        try:
            x, y, z = (float(t) for t in v.split())
            return (x, y, z)
        except ValueError:
            return None

    @property
    def text_origin(self):
        return self.vec("textOrigin")

    @property
    def leader_terminator(self):
        return self.vec("Leader[0].terminator")

    @property
    def reference(self) -> str:
        """Raw NX associativity reference of the first leader, if any."""
        return self.props.get("Leader[0].Reference", "")

    def datum_refs(self, compartment: int = 0) -> list[list[str]]:
        """[[primary labels], [secondary labels], [tertiary labels]]."""
        out = []
        for level in ("Primary", "Secondary", "Tertiary"):
            labels = []
            m = 0
            while True:
                k = f"ToleranceCompartment[{compartment}].{level}Datum.Reference[{m}].label"
                if k not in self.props:
                    break
                labels.append(self.props[k])
                m += 1
            out.append(labels)
        return out


@dataclass
class JtPmi:
    path: str
    version: str
    entities: list[JtPmiEntity] = field(default_factory=list)
    # Model-view counters etc. (CAD_PMI_COUNT_*, ModelViewCameraType ...)
    summary: dict[str, str] = field(default_factory=dict)
    # id -> raw name from the PMI index ("Feature Control Frame (35)")
    names: dict[int, str] = field(default_factory=dict)

    def of_kind(self, kind: str) -> list[JtPmiEntity]:
        return [e for e in self.entities if e.kind == kind]


def _mbs(b: bytes, p: int):
    if p + 4 > len(b):
        return None
    n = struct.unpack_from("<i", b, p)[0]
    if n < 0 or n > 4000 or p + 4 + 2 * n > len(b):
        return None
    try:
        s = b[p + 4:p + 4 + 2 * n].decode("utf-16le")
    except UnicodeDecodeError:
        return None
    if not s.isprintable():
        return None
    return s, p + 4 + 2 * n


def _pair(b: bytes, p: int):
    k = _mbs(b, p)
    if not k or not k[0] or not _KEY_RE.match(k[0]) or k[1] >= len(b):
        return None
    v = _mbs(b, k[1] + 1)
    if v is None:
        return None
    return k[0], v[0], v[1] + 1  # +1 trailing flag byte


def _scan_runs(b: bytes):
    """Yield (start_offset, [(key, value), ...]) for each maximal pair run."""
    p = 0
    n = len(b)
    while p < n:
        if not _pair(b, p):
            p += 1
            continue
        start = p
        items = []
        while True:
            r = _pair(b, p)
            if not r:
                break
            items.append((r[0], r[1]))
            p = r[2]
        yield start, items


def _kind_from(name: str, props: dict[str, str]) -> str:
    m = _NAME_RE.match(name)
    low = (m.group(1) if m else name).lower()
    if "feature control frame" in low:
        return KIND_FCF
    if "datum feature symbol" in low:
        return KIND_DATUM
    if "datum target" in low:
        return KIND_TARGET
    if "dimension" in low:
        return KIND_DIM
    if low.startswith("note") or "balloon" in low or low.startswith("label"):
        return KIND_NOTE
    if "surface finish" in low or "roughness" in low:
        return KIND_FINISH
    # NX gives no per-entity name in the property bag (the id is an object
    # id, not the "(35)" display number), so classify by key signature.
    if any(k.startswith("ToleranceCompartment") for k in props):
        return KIND_FCF
    if "ValueText" in props or "ValueText.Item[0].string" in props or "upperDelta" in props:
        return KIND_DIM
    if "label" in props and "Leader[0]" in props:
        # Datum targets (A1..An) carry an ``index``; feature symbols do not.
        return KIND_TARGET if "index" in props else KIND_DATUM
    if any(k.startswith("Text[0]") for k in props):
        return KIND_NOTE
    if "SurfaceFinish" in "".join(props) or "roughness" in "".join(props).lower():
        return KIND_FINISH
    if "textOrigin" in props and "Outline" in props:
        return KIND_SYMBOL
    return KIND_OTHER


def _collect_names(b: bytes) -> dict[int, str]:
    """All "<display name> (<id>)" strings in the index part of the blob."""
    names: dict[int, str] = {}
    p = 0
    while p < len(b) - 4:
        r = _mbs(b, p)
        if r and r[0]:
            m = _NAME_RE.match(r[0])
            if m and m.group(1):
                names.setdefault(int(m.group(2)), r[0])
            p = r[1]
        else:
            p += 1
    return names


_MARK_TOL = "Tolerance".encode("utf-16le")
_MARK_CNT = "CAD_PMI_COUNT".encode("utf-16le")


def extract_pmi(jt: JtFile) -> JtPmi:
    pmi = JtPmi(jt.path, jt.version)
    seen: set[tuple[int, str]] = set()
    for seg in jt.segments_of(SEG_PMI):
        try:
            blob = jt.payload(seg)
        except Exception:  # noqa: BLE001 - corrupt / unsupported compression
            continue
        # Only the PMI-manager element is a big property bag; the small
        # "Multicad Geometry" segments are drawn geometry (names only).
        if _MARK_TOL not in blob and _MARK_CNT not in blob:
            continue
        names = _collect_names(blob)
        pmi.names.update(names)
        for start, items in _scan_runs(blob):
            if start < 8:
                continue
            ident, count = struct.unpack_from("<ii", blob, start - 8)
            if count != len(items):
                continue  # not an entity record header
            props = dict(items)
            if any(k.startswith("CAD_PMI_COUNT") for k in props):
                pmi.summary.update(props)
                continue
            if ident <= 0:
                continue
            kind = _kind_from(names.get(ident, ""), props)
            if kind == KIND_OTHER:
                continue  # view settings, reference points, etc.
            key = (ident, kind)
            if key in seen:
                continue
            seen.add(key)
            pmi.entities.append(
                JtPmiEntity(ident, kind, names.get(ident, ""), props, seg.offset, start)
            )
    pmi.entities.sort(key=lambda e: e.id)
    return pmi
