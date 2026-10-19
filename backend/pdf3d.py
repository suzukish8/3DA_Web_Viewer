"""3D PDF(U3D 埋め込み)の書き出し。

モデルの三角形メッシュと、注記(線・塗りつぶし)を 1 つの U3D にして、Acrobat Reader で回転・拡大できる PDF に埋め込む。
- 線は細い三角柱にする(U3D のメッシュとして書くため)。色は頂点ごと。
- U3D への変換は pymeshlab(MeshLab 同梱の IDTFConverter)を使う。
"""
from __future__ import annotations

import math
import os
import tempfile
import zlib

import numpy as np


def _tubes(polylines, half):
    """折れ線(各 (N,3))を、細い三角柱の頂点・面にする。戻り値: (頂点 (V,3), 面 (F,3))"""
    A, B = [], []
    for pl in polylines:
        pl = np.asarray(pl, float).reshape(-1, 3)
        if len(pl) < 2:
            continue
        A.append(pl[:-1])
        B.append(pl[1:])
    if not A:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    A = np.vstack(A)
    B = np.vstack(B)
    d = B - A
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-9
    A, B, d, L = A[ok], B[ok], d[ok], L[ok]
    if not len(A):
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    d = d / L[:, None]
    ref = np.where((np.abs(d[:, 2]) < 0.9)[:, None], np.array([[0, 0, 1.0]]), np.array([[0, 1.0, 0]]))
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    ring = [math.cos(t) * u + math.sin(t) * v for t in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)]
    S = len(A)
    verts = np.stack([A + half * r for r in ring] + [B + half * r for r in ring], axis=1).reshape(-1, 3)   # (S*6, 3)
    b0 = np.arange(S) * 6
    f = []
    for k in range(3):
        k2 = (k + 1) % 3
        f.append(np.stack([b0 + k, b0 + k2, b0 + 3 + k], axis=1))
        f.append(np.stack([b0 + k2, b0 + 3 + k2, b0 + 3 + k], axis=1))
    return verts, np.vstack(f)


def build_u3d(positions, indices, model_rgb, lines, solids, diag, line_width_mm=None):
    """positions: (V,3) mm / indices: (F,3) / lines: [{rgb, pts}] / solids: [{rgb, pts}] -> U3D のバイト列"""
    import pymeshlab as ml
    half = (line_width_mm / 2) if line_width_mm else max(diag * 0.0006, 0.03)
    V = [np.asarray(positions, float).reshape(-1, 3)]
    F = [np.asarray(indices, np.int64).reshape(-1, 3)]
    C = [np.tile(np.array(model_rgb, float) / 255.0, (len(V[0]), 1))]
    off = len(V[0])
    by = {}
    for ln in lines:                                           # 線: 色ごとにまとめて三角柱にする
        by.setdefault(tuple(int(x) for x in ln["rgb"]), []).append(np.asarray(ln["pts"], float).reshape(-1, 3))
    for rgb, pls in by.items():
        vv, ff = _tubes(pls, half)
        if not len(vv):
            continue
        V.append(vv)
        F.append(ff + off)
        C.append(np.tile(np.array(rgb, float) / 255.0, (len(vv), 1)))
        off += len(vv)
    by = {}
    for s in solids:                                           # 塗りつぶし: 扇形に三角形へ。両面から見えるよう、裏向きの面も足す
        by.setdefault(tuple(int(x) for x in s["rgb"]), []).append(np.asarray(s["pts"], float).reshape(-1, 3))
    for rgb, pls in by.items():
        vs, fs, o = [], [], 0
        for p in pls:
            if len(p) < 3:
                continue
            vs.append(p)
            fs.append(np.array([[o, o + i, o + i + 1] for i in range(1, len(p) - 1)]))
            o += len(p)
        if not vs:
            continue
        vv = np.vstack(vs)
        ff = np.vstack(fs)
        ff = np.vstack([ff, ff[:, ::-1]])
        V.append(vv)
        F.append(ff + off)
        C.append(np.tile(np.array(rgb, float) / 255.0, (len(vv), 1)))
        off += len(vv)
    verts = np.vstack(V)
    faces = np.vstack(F)
    # Acrobat で開くと、MeshLab 経由の U3D の形は (x, y, z) -> (-x, z, y) に回って表示される(実測)。逆に回してから渡して、元の座標に戻す
    verts = np.stack([-verts[:, 0], verts[:, 2], verts[:, 1]], axis=1)
    col = np.vstack(C)
    col = np.hstack([col, np.ones((len(col), 1))])
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(verts, faces, v_color_matrix=col))
    fd, path = tempfile.mkstemp(suffix=".u3d")
    os.close(fd)
    try:
        ms.save_current_mesh(path, save_vertex_color=True, save_face_color=False)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def write_pdf3d(path, u3d, center, right, up, n, radius, page_wh=(842.0, 595.0), cam=None):
    """U3D を埋め込んだ PDF を書く。カメラは (right, up, n) の向きで center を見る(n = 見る人の側)"""
    W, H = page_wh
    fov = 30.0
    dist = 1.25 * radius / math.tan(math.radians(fov / 2))
    if cam is None:
        cam = np.asarray(center, float) + np.asarray(n, float) * dist
    # Acrobat のカメラは +Z 方向を見る(画面の右 = -X 軸)。見たい向き (right, up, n) に合わせて X = -right, Y = up, Z = -n にする
    r_, u_, n_ = np.asarray(right, float), np.asarray(up, float), np.asarray(n, float)
    c2w = [float(x) for x in -r_] + [float(x) for x in u_] + [float(x) for x in -n_] + [float(x) for x in cam]
    arr = " ".join(f"{x:.6f}" for x in c2w)
    objs = []

    def add(b):
        objs.append(b if isinstance(b, bytes) else b.encode("latin-1"))
        return len(objs)

    comp = zlib.compress(u3d)
    # 1 Catalog, 2 Pages, 3 Page, 4 Annot(3D), 5 3D stream, 6 View, 7 Contents
    add("<< /Type /Catalog /Pages 2 0 R >>")
    add("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    m = 24
    add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] /Annots [4 0 R] /Contents 7 0 R >>")
    add(f"<< /Type /Annot /Subtype /3D /Rect [{m} {m} {W - m} {H - m}] /F 68 /3DD 5 0 R /3DV 6 0 R "
        f"/3DA << /A /PO /AIS /L /D /PC /DIS /U /TB true /NP false >> /3DI true /AP << /N 8 0 R >> >>")
    add(b"<< /Type /3D /Subtype /U3D /Filter /FlateDecode /Length %d >>\nstream\n" % len(comp) + comp + b"\nendstream")
    add(f"<< /Type /3DView /XN (Default) /IN (Default) /MS /M /C2W [{arr}] /CO {dist:.6f} "
        f"/P << /Type /3DProjection /Subtype /P /PS /Min /FOV {fov} >> "
        f"/BG << /Type /3DBG /Subtype /SC /CS /DeviceRGB /C [1 1 1] >> "
        f"/RM << /Type /3DRenderMode /Subtype /SolidOutline >> /LS << /Type /3DLightingScheme /Subtype /CAD >> >>")
    add(b"<< /Length 0 >>\nstream\n\nendstream")
    # 8: 3D が有効になる前に出る表示(Acrobat 既定の「?」の代わり)
    bw, bh = W - 2 * m, H - 2 * m
    msg = "3D content - click here to activate. (If it does not appear, allow 3D content in Acrobat.)"
    cs = (f"0.96 0.96 0.97 rg 0 0 {bw} {bh} re f 0.7 0.7 0.75 RG 1 w 0.5 0.5 {bw - 1} {bh - 1} re S "
          f"BT /F1 14 Tf 0.3 0.3 0.35 rg {bw / 2 - 270} {bh / 2 - 5} Td ({msg}) Tj ET").encode("latin-1")
    add(f"<< /Type /XObject /Subtype /Form /BBox [0 0 {bw} {bh}] /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /Length {len(cs)} >>\nstream\n".encode("latin-1") + cs + b"\nendstream")
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offs = []
    for i, b in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + b + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
    with open(path, "wb") as f:
        f.write(out)
