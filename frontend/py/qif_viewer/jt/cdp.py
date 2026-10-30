"""JT Int32 Compressed Data Packet (CDP) decoder -- spec 10.2.1 / 10.3 / Annex B.

``Stream`` is a little-endian byte cursor; ``read_cdp`` decodes one Int32CDP
(Null / Bitlength / Arithmetic / Chopper / Move-to-front) into a list of ints.
Bit fields (probability contexts, CodeText) are read MSB-first from
consecutive little-endian U32 words.
"""
from __future__ import annotations

import struct

NIBBLE = 4  # BitLength nibbler width


class Stream:
    def __init__(self, data: bytes, pos: int = 0):
        self.d = data
        self.p = pos

    def u8(self) -> int:
        v = self.d[self.p]
        self.p += 1
        return v

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.d, self.p)[0]
        self.p += 8
        return v

    def f32(self) -> float:
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def words(self, n: int) -> list[int]:
        w = list(struct.unpack_from(f"<{n}I", self.d, self.p))
        self.p += 4 * n
        return w


class BitReader:
    """MSB-first reader over a list of U32 words."""

    def __init__(self, words: list[int], nbits: int | None = None):
        self.w = words
        self.pos = 0  # absolute bit position
        self.nbits = nbits if nbits is not None else 32 * len(words)

    def get(self, n: int) -> int:
        if n == 0:
            return 0
        v = 0
        pos = self.pos
        need = n
        while need > 0:
            wi, bi = divmod(pos, 32)
            word = self.w[wi] if wi < len(self.w) else 0
            take = min(need, 32 - bi)
            v = (v << take) | ((word >> (32 - bi - take)) & ((1 << take) - 1))
            pos += take
            need -= take
        self.pos = pos
        return v

    def get_signed(self, n: int) -> int:
        v = self.get(n)
        if n and v >= 1 << (n - 1):
            v -= 1 << n
        return v

    def align(self) -> None:
        self.pos = (self.pos + 7) & ~7

    def nibbler(self, signed: bool) -> int:
        val, k = 0, 0
        while True:
            val |= self.get(NIBBLE) << (k * NIBBLE)
            more = self.get(1)
            k += 1
            if not more:
                break
        if signed:
            sw = k * NIBBLE
            if val >= 1 << (sw - 1):
                val -= 1 << sw
        return val


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def _bitlength_decode(words: list[int], nbits: int, n: int) -> list[int]:
    br = BitReader(words, nbits)
    out: list[int] = []
    if br.get(1) == 0:  # fixed width
        mn = br.nibbler(True)
        mx = br.nibbler(True)
        width = max(0, (mx - mn)).bit_length()
        for _ in range(n):
            out.append(br.get(width) + mn)
        return out
    mean = br.nibbler(True)
    blk_len_bits, blk_val_bits = 4, 4
    max_decr = -(1 << (blk_val_bits - 1))
    max_incr = (1 << (blk_val_bits - 1)) - 1
    cur = 0
    while len(out) < n:
        while True:
            d = br.get_signed(blk_val_bits)
            cur += d
            if d != max_decr and d != max_incr:
                break
        run = br.get(blk_len_bits)
        for _ in range(run):
            out.append(br.get_signed(cur) + mean)
    return out[:n]


class ByteBits:
    """MSB-first bit reader over raw bytes (probability context tables)."""

    def __init__(self, data: bytes, pos: int):
        self.d = data
        self.pos = pos * 8

    def get(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.d[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def byte_pos(self) -> int:
        return (self.pos + 7) // 8


class _Ctx:
    """Int32 probability context table (entries stay in file order)."""

    def __init__(self, br: ByteBits, value_bits: int | None):
        cnt = br.get(16)
        self.occ_bits = br.get(6)
        if value_bits is None:
            value_bits = br.get(7)
        self.value_bits = value_bits
        self.min_value = br.get(32)
        self.entries: list[tuple[bool, int, int, int]] = []  # esc,count,cum,value
        cum = 0
        for _ in range(cnt):
            esc = br.get(1) == 1
            c = br.get(self.occ_bits)
            v = br.get(value_bits) + _s32(self.min_value)
            self.entries.append((esc, c, cum, v))
            cum += c
        self.total = cum

    def lookup(self, code: int):
        lo, hi = 0, len(self.entries) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            e = self.entries[mid]
            if code < e[2]:
                hi = mid - 1
            elif code >= e[2] + e[1]:
                lo = mid + 1
            else:
                return e
        return self.entries[lo]


def _arith_decode(words: list[int], nbits: int, n: int, ctx: _Ctx, oob: list[int]) -> list[int]:
    w = words
    bitbuf = 0
    nbuf = 0
    wi = 0

    def next_word():
        nonlocal wi, bitbuf, nbuf
        bitbuf = w[wi] if wi < len(w) else 0
        nbuf = min(32, max(0, nbits - 32 * wi)) if nbits else 32
        if wi >= len(w):
            nbuf = 32
        wi += 1

    def read_bit() -> int:
        nonlocal bitbuf, nbuf
        if nbuf == 0:
            next_word()
        b = (bitbuf >> 31) & 1
        bitbuf = (bitbuf << 1) & 0xFFFFFFFF
        nbuf -= 1
        return b

    next_word()
    low, high = 0, 0xFFFF
    code = bitbuf >> 16
    bitbuf = (bitbuf << 16) & 0xFFFFFFFF
    nbuf -= 16
    out: list[int] = []
    oi = 0
    total = ctx.total
    for _ in range(n):
        rng = high - low + 1
        rescaled = (((code - low) + 1) * total - 1) // rng
        e = ctx.lookup(rescaled)
        if not e[0]:
            out.append(e[3])
        else:
            out.append(oob[oi] if oi < len(oob) else 0)
            oi += 1
        lo_ct, hi_ct = e[2], e[2] + e[1]
        high = low + (rng * hi_ct) // total - 1
        low = low + (rng * lo_ct) // total
        while True:
            if ((~(high ^ low)) & 0xFFFF) >> 15:
                pass
            elif (low >> 14) == 1 and (high >> 14) == 2:
                code ^= 0x4000
                low &= 0x3FFF
                high |= 0x4000
            else:
                break
            low = (low << 1) & 0xFFFF
            high = ((high << 1) | 1) & 0xFFFF
            code = (code << 1) & 0xFFFF
            code |= read_bit()
    return out


def read_cdp(s: Stream, ext_compressed: bool = False, depth: int = 0) -> list[int]:
    """Decode one Int32 Compressed Data Packet from ``s``."""
    n = s.i32()
    if n <= 0:
        return []
    codec = s.u8()
    if codec in (0, 1):
        nbits = s.i32()
        words = s.words((nbits + 31) // 32)
        if codec == 0:
            return [_s32(x) for x in words[:n]]
        return _bitlength_decode(words, nbits, n)
    if codec == 3:
        nbits = s.i32()
        words = s.words((nbits + 31) // 32)
        br = ByteBits(s.d, s.p)
        ctx = _Ctx(br, None)
        s.p = br.byte_pos()
        oob: list[int] = []
        if any(e[0] for e in ctx.entries):  # out-of-band values only exist with an escape symbol
            if ext_compressed:
                noob = s.i32()
                oob = [_s32(x) for x in s.words(noob)]
            else:
                oob = read_cdp(s, ext_compressed, depth + 1)
        if nbits == 0:
            return oob[:n]
        return _arith_decode(words, nbits, n, ctx, oob)
    if codec == 4:
        chop = s.u8()
        bias = s.i32()
        span = s.u8()
        msb = read_cdp(s, ext_compressed, depth + 1)
        lsb = read_cdp(s, ext_compressed, depth + 1)
        sh = span - chop
        return [_s32(((l & ((1 << sh) - 1)) | (m << sh)) + bias) for m, l in zip(msb, lsb)]
    if codec == 5:
        vals = read_cdp(s, ext_compressed, depth + 1)
        offs = read_cdp(s, ext_compressed, depth + 1)
        window: list[int] = []
        vi = 0
        out = []
        for o in offs:
            if o == -1 or o >= len(window):
                v = vals[vi]
                vi += 1
            else:
                v = window.pop(o)
            window.insert(0, v)
            del window[16:]
            out.append(v)
        return out[:n]
    raise ValueError(f"unknown CDP codec {codec}")


def unpredict_lag1(res: list[int]) -> list[int]:
    """Lag1: the first four values are primers, then value = residual + previous."""
    out = [r & 0xFFFFFFFF for r in res[:4]]
    for r in res[4:]:
        out.append((out[-1] + r) & 0xFFFFFFFF)
    return out


def unpredict_xor1(res: list[int]) -> list[int]:
    out = [r & 0xFFFFFFFF for r in res[:4]]
    for r in res[4:]:
        out.append((out[-1] ^ r) & 0xFFFFFFFF)
    return out
