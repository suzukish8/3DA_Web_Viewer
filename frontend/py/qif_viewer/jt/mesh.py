"""JT Tri-Strip Set Shape LOD decoder (spec 6.1.1, 10.2.3, Annex D).

``decode_lod_segment`` returns (vertices, triangles) for one LOD segment
payload (segment header already stripped).  Only vertex coordinates and the
triangle topology are decoded; normals / colours / texture coordinates are
not needed by the viewer (it recomputes normals).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .cdp import Stream, read_cdp, unpredict_lag1

TRISTRIP_LOD_GUID = bytes.fromhex("ab10dd10c82ad1119b6b0080c7bb5997")


# ---------------------------------------------------------------- dual mesh
class _Dual:
    """Dual vertex-face mesh built by the topology decoder."""

    def __init__(self) -> None:
        self.vcnt: list[int] = []          # valence per vertex
        self.vgrp: list[int] = []
        self.vflag: list[int] = []
        self.vface: list[list[int]] = []   # incident faces per vertex
        self.fdeg: list[int] = []
        self.fvtx: list[list[int]] = []    # incident vertices per face
        self.fempty: list[int] = []

    def new_vtx(self, val: int) -> int:
        self.vcnt.append(val)
        self.vgrp.append(-1)
        self.vflag.append(0)
        self.vface.append([-1] * val)
        return len(self.vcnt) - 1

    def new_face(self, deg: int) -> int:
        self.fdeg.append(deg)
        self.fvtx.append([-1] * deg)
        self.fempty.append(deg)
        return len(self.fdeg) - 1

    def valid_face(self, f: int) -> bool:
        return 0 <= f < len(self.fdeg) and self.fdeg[f] != 0

    def set_face_vtx(self, f: int, slot: int, v: int) -> None:
        lst = self.fvtx[f]
        if lst[slot] != v:
            self.fempty[f] -= 1
        lst[slot] = v

    def find_vtx_slot(self, f: int, v: int) -> int:
        try:
            return self.fvtx[f].index(v)
        except ValueError:
            return -1

    def find_face_slot(self, v: int, f: int) -> int:
        try:
            return self.vface[v].index(f)
        except ValueError:
            return -1


class MeshError(Exception):
    pass


class _Decoder:
    def __init__(self, deg_syms, val_syms, grp_syms, flag_syms, split_face, split_pos):
        self.deg = deg_syms          # list of 8 lists
        self.deg_pos = [0] * 8
        self.val = val_syms
        self.val_pos = 0
        self.grp = grp_syms
        self.grp_pos = 0
        self.flag = flag_syms
        self.flag_pos = 0
        self.sface = split_face
        self.sface_pos = 0
        self.spos = split_pos
        self.spos_pos = 0
        self.m = _Dual()
        self.active: list[int] = []
        self.removed: set[int] = set()

    # --- symbol streams
    def _next(self, arr, attr):
        i = getattr(self, attr)
        if i < len(arr):
            setattr(self, attr, i + 1)
            return arr[i]
        return -1

    def _face_cntxt(self, v: int) -> int:
        m = self.m
        cval = m.vcnt[v]
        known = tot = 0
        for f in m.vface[v]:
            if not m.valid_face(f):
                continue
            known += 1
            tot += m.fdeg[f]
        if cval == 3:
            return 0 if tot < known * 6 else (1 if tot == known * 6 else 2)
        if cval == 4:
            return 3 if tot < known * 4 else (4 if tot == known * 4 else 5)
        if cval == 5:
            return 6
        return 7

    # --- I/O (MeshDecoder)
    def io_vtx(self) -> int:
        sym = self._next(self.val, "val_pos")
        if sym <= -1:
            return -1
        v = self.m.new_vtx(sym)
        self.m.vgrp[v] = self._next(self.grp, "grp_pos")
        f = self._next(self.flag, "flag_pos")
        self.m.vflag[v] = max(f, 0)
        return v

    def io_face(self, v: int) -> int:
        c = self._face_cntxt(v)
        i = self.deg_pos[c]
        if i >= len(self.deg[c]):
            return -2
        self.deg_pos[c] = i + 1
        sym = self.deg[c][i]
        if sym == 0:
            return -1
        if sym < 0:
            return -2
        return self.m.new_face(sym)

    def io_split_face(self) -> int:
        sym = self._next(self.sface, "sface_pos")
        if sym > -1:
            n = len(self.active)
            if sym <= 0 or sym > n:
                return -2
            return self.active[n - sym]
        return -1

    def io_split_pos(self) -> int:
        return self._next(self.spos, "spos_pos")

    # --- MeshCodec
    def run(self) -> _Dual:
        while self.run_component():
            pass
        return self.m

    def run_component(self) -> bool:
        v = self.io_vtx()
        if v == -1:
            return False
        for i in range(self.m.vcnt[v]):
            if self.activate_f(v, i) == -2:
                raise MeshError("mesh traversal failed")
        while True:
            f = self.next_active_face()
            if f == -1:
                break
            self.complete_face(f)
            self.removed.add(f)
        return True

    def complete_face(self, f: int) -> None:
        m = self.m
        while True:
            j = m.find_vtx_slot(f, -1)
            if j == -1:
                break
            v = self.activate_v(f, j)
            self.complete_vtx(v, j)

    def activate_v(self, f: int, slot: int) -> int:
        v = self.io_vtx()
        if v < 0:
            raise MeshError("vertex symbols exhausted")
        self.m.vface[v][0] = f
        self.add_vtx_to_face(v, 0, f, slot)
        return v

    def activate_f(self, v: int, slot: int) -> int:
        m = self.m
        f = self.io_face(v)
        if f >= 0:
            m.vface[v][slot] = f
            m.set_face_vtx(f, 0, v)
            self.active.append(f)
        elif f == -1:
            f = self.io_split_face()
            j = self.io_split_pos()
            if f == -2 or slot == -1:
                return -2
            m.vface[v][slot] = f
            self.add_vtx_to_face(v, slot, f, j)
        return f

    def complete_vtx(self, v: int, vslot: int) -> None:
        m = self.m
        cval = m.vcnt[v]
        vp = m.vface[v][0]
        jp = vslot
        i = 1
        while i < cval and (vn := m.vface[v][i]) != -1:
            jp = (jp - 1) % m.fdeg[vp]
            v2 = m.fvtx[vp][jp]
            if v2 == -1:
                break
            jn = m.find_vtx_slot(vn, v2)
            if jn < 0:
                raise MeshError("ccw link failed")
            jn = (jn - 1) % m.fdeg[vn]
            self.add_vtx_to_face(v, i, vn, jn)
            vp, jp = vn, jn
            i += 1
            if i >= cval:
                return
        ilast = i
        vp = m.vface[v][0]
        jp = vslot
        i = cval - 1
        while i >= 0 and (vn := m.vface[v][i]) != -1:
            jp = (jp + 1) % m.fdeg[vp]
            v2 = m.fvtx[vp][jp]
            if v2 == -1:
                break
            jn = m.find_vtx_slot(vn, v2)
            if jn < 0:
                raise MeshError("cw link failed")
            jn = (jn + 1) % m.fdeg[vn]
            self.add_vtx_to_face(v, i, vn, jn)
            vp, jp = vn, jn
            i -= 1
            if i < ilast:
                return
        while ilast <= i:
            if self.activate_f(v, ilast) < -1:
                raise MeshError("activate failed")
            ilast += 1

    def add_vtx_to_face(self, v: int, jf: int, f: int, vslot: int) -> None:
        m = self.m
        deg = m.fdeg[f]
        ccw = (vslot + 1) % deg
        cw = (vslot - 1) % deg
        m.set_face_vtx(f, vslot, v)
        fp = m.fvtx[f][cw]
        if fp != -1:
            ip = m.find_face_slot(fp, f)
            s = (jf + 1) % m.vcnt[v]
            if m.vface[v][s] == -1:
                ip = (ip - 1) % m.vcnt[fp]
                m.vface[v][s] = m.vface[fp][ip]
        fn = m.fvtx[f][ccw]
        if fn != -1:
            inn = m.find_face_slot(fn, f)
            s = (jf - 1) % m.vcnt[v]
            if m.vface[v][s] == -1:
                inn = (inn + 1) % m.vcnt[fn]
                m.vface[v][s] = m.vface[fn][inn]

    def next_active_face(self) -> int:
        act, rem = self.active, self.removed
        while act and act[-1] in rem:
            act.pop()
        best = -1
        lowest = 9999999
        i = len(act) - 1
        while i >= max(0, len(act) - 16):
            f0 = act[i]
            if f0 in rem:
                del act[i]
                i -= 1
                continue
            e = self.m.fempty[f0]
            if e < lowest:
                lowest = e
                best = f0
            i -= 1
        return best


# ------------------------------------------------------------- element parse
@dataclass
class LodMesh:
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    groups: list[int] = field(default_factory=list)  # face-group id per triangle


def _dequantize(codes: list[int], lo: float, hi: float, bits: int) -> list[float]:
    maxcode = (1 << bits) - 1 if bits < 32 else 0xFFFFFFFF
    k = (hi - lo) / maxcode if maxcode else 0.0
    return [lo + c * k for c in codes]


def decode_tristrip_lod(body: bytes) -> LodMesh:
    """``body`` = segment bytes after the 24-byte segment header."""
    s = Stream(body, 0)
    s.i32()                     # element length
    guid = body[4:20]
    s.p = 20
    if guid != TRISTRIP_LOD_GUID:
        raise MeshError("not a tri-strip LOD element")
    s.u8()                      # base type
    s.i32()                     # object id
    s.u8()                      # vertex shape LOD version
    s.u8()                      # base shape LOD version
    bindings = s.u64()
    s.i32()                     # length of the nested topology element
    s.p += 16                   # its GUID
    s.u8()
    s.i32()
    s.u8()                      # TopoMesh LOD data version
    s.u32()                     # vertex records object id
    s.u8()                      # topologically compressed LOD version

    deg = [read_cdp(s) for _ in range(8)]
    val = read_cdp(s)
    grp = read_cdp(s)
    flags = unpredict_lag1(read_cdp(s))
    masks = [read_cdp(s) for _ in range(8)]  # face attribute masks (32 LSB), unused
    if masks[7]:
        read_cdp(s)             # context-7 masks continue with their 32 MSBs
    # Optional mask-MSB / high-degree-mask arrays may or may not be present
    # (they are empty or absent depending on the face degrees); the split
    # arrays are the ones whose length equals the number of split (degree 0)
    # symbols, so locate them by that count.
    nsplit = sum(d.count(0) for d in deg)

    def peek_count() -> int:
        return struct.unpack_from("<i", s.d, s.p)[0]

    if nsplit > 0:
        while peek_count() != nsplit:
            read_cdp(s)
        sface = unpredict_lag1(read_cdp(s))
        spos = read_cdp(s)
    else:
        while peek_count() == 0:
            read_cdp(s)
        sface, spos = [], []
    sface = [_to_i32(x) for x in sface]
    s.u32()                     # composite hash

    dual = _Decoder(deg, val, grp, [_to_i32(x) for x in flags], sface, spos).run()

    # ---- vertex records
    vb = s.u64()
    s.u8(), s.u8(), s.u8(), s.u8()   # quantization params
    ntopo = s.i32()
    if ntopo <= 0:
        raise MeshError("no vertices")
    s.i32()                     # number of vertex attributes
    nuniq = s.i32()
    ncomp = s.u8()
    qs = []
    for _ in range(3):
        lo, hi, bits = s.f32(), s.f32(), s.u8()
        qs.append((lo, hi, bits))
    comps = []
    for c in range(ncomp):
        codes = unpredict_lag1(read_cdp(s))
        lo, hi, bits = qs[c]
        if bits == 0:
            comps.append([struct.unpack("<f", struct.pack("<I", x & 0xFFFFFFFF))[0] for x in codes])
        else:
            comps.append(_dequantize(codes, lo, hi, bits))
    verts = list(zip(*comps))
    if len(verts) != len(dual.fdeg):
        raise MeshError(f"vertex count {len(verts)} != dual faces {len(dual.fdeg)}")

    mesh = LodMesh(vertices=verts)
    for v in range(len(dual.vcnt)):
        if dual.vflag[v] != 0 or dual.vcnt[v] != 3:
            continue  # cover faces (flag 1) only close open boundaries
        a, b, c = dual.vface[v]
        if min(a, b, c) < 0:
            continue
        mesh.triangles.append((a, b, c))
        mesh.groups.append(dual.vgrp[v])
    return mesh


def _to_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v
