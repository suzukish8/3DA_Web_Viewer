"""File > Export Current View (DXF / SVG / PDF): 3DA B&M Viewer の書き出し処理(geometry/dxf_export.py など)をそのまま使う。
フロントエンドが、現在の画面を 2D(mm、Y 上向き)に平らにした図形を JSON で送り、ここでファイルにして返す。"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.environ.get("QIF_VIEWER_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from fastapi import APIRouter, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

from qif_viewer.geometry.dxf_export import DxfCircle, DxfLabel, DxfPolyline, DxfSolid, write_dxf  # noqa: E402
from qif_viewer.geometry.pdf_export import drawing_scale_value, page_size_pt, write_pdf  # noqa: E402
from qif_viewer.geometry.svg_export import write_svg  # noqa: E402

router = APIRouter()

_MEDIA = {"dxf": "application/dxf", "svg": "image/svg+xml", "pdf": "application/pdf"}


def _rgb(v) -> tuple[int, int, int]:
    try:
        return (int(v[0]), int(v[1]), int(v[2]))
    except Exception:
        return (0, 0, 0)


@router.post("/api/export/{fmt}")
async def export_view(fmt: str, body: dict):
    fmt = fmt.lower()
    if fmt not in _MEDIA:
        raise HTTPException(status_code=400, detail="未対応の形式です")
    polylines = [
        DxfPolyline(layer=str(p.get("layer", "0")), rgb=_rgb(p.get("rgb")), points=np.asarray(p["points"], dtype=float).reshape(-1, 2))
        for p in body.get("polylines", []) if len(p.get("points", [])) >= 2
    ]
    circles = [
        DxfCircle(layer=str(c.get("layer", "0")), rgb=_rgb(c.get("rgb")), center=(float(c["c"][0]), float(c["c"][1])), radius=float(c["r"]))
        for c in body.get("circles", [])
    ]
    solids = []
    for s in body.get("solids", []):
        pts = np.asarray(s.get("points", []), dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            continue
        layer, rgb = str(s.get("layer", "0")), _rgb(s.get("rgb"))
        if fmt == "dxf" and len(pts) > 3:        # DXF の SOLID は三角形(4 点)まで。黒丸などの多角形は三角形に分けて塗りつぶす
            for i in range(1, len(pts) - 1):
                solids.append(DxfSolid(layer=layer, rgb=rgb, points=np.array([pts[0], pts[i], pts[i + 1]])))
        else:
            solids.append(DxfSolid(layer=layer, rgb=rgb, points=pts))
    labels = [
        DxfLabel(layer=str(t.get("layer", "0")), rgb=_rgb(t.get("rgb")), position=(float(t["p"][0]), float(t["p"][1])),
                 text=str(t["text"]), height=float(t["h"]))
        for t in body.get("labels", [])
    ]
    widths = body.get("line_widths") or [0.25, 0.125]
    line_widths = (float(widths[0]), float(widths[1]))

    fd, path = tempfile.mkstemp(suffix="." + fmt)
    os.close(fd)
    try:
        if fmt == "dxf":
            write_dxf(path, polylines, circles, labels, solids, line_widths=line_widths)
        elif fmt == "svg":
            write_svg(path, polylines, circles, labels, solids)
        else:
            write_pdf(path, polylines, circles, labels, solids,
                      page_size=page_size_pt(body.get("page_size")),
                      scale=drawing_scale_value(body.get("scale")), line_widths=line_widths)
    except Exception as e:  # noqa: BLE001
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"書き出しに失敗しました: {e}")
    name = (body.get("filename") or "view") + "." + fmt
    return FileResponse(path, media_type=_MEDIA[fmt], filename=name,
                        background=BackgroundTask(lambda: os.path.exists(path) and os.remove(path)))


@router.post("/api/export3dpdf")
async def export_3dpdf(body: dict):
    """3D PDF: モデルのメッシュ + 注記(3D の線・塗りつぶし)を U3D にして PDF に埋め込む"""
    import asyncio
    import base64

    from pdf3d import build_u3d, write_pdf3d
    try:
        pos = np.frombuffer(base64.b64decode(body["positions"]), dtype="<f4").astype(float).reshape(-1, 3)
        idx = np.frombuffer(base64.b64decode(body["indices"]), dtype="<u4").astype(np.int64).reshape(-1, 3)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"メッシュを読めません: {e}")
    if not len(pos) or not len(idx):
        raise HTTPException(status_code=400, detail="メッシュがありません")
    lines = [{"rgb": _rgb(x.get("rgb")), "pts": x["pts"]} for x in body.get("lines", []) if len(x.get("pts", [])) >= 6]
    solids = [{"rgb": _rgb(x.get("rgb")), "pts": x["pts"]} for x in body.get("solids", []) if len(x.get("pts", [])) >= 9]
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        u3d = await asyncio.to_thread(build_u3d, pos, idx, _rgb(body.get("model_rgb") or [176, 182, 192]), lines, solids, diag, body.get("line_mm"))
        v = body.get("view") or {}
        write_pdf3d(path, u3d, (lo + hi) / 2, v.get("right", [1, 0, 0]), v.get("up", [0, 1, 0]), v.get("n", [0, 0, 1]), diag / 2)
    except Exception as e:  # noqa: BLE001
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"3D PDF を作れませんでした: {e}")
    return FileResponse(path, media_type="application/pdf", filename=str(body.get("filename", "model")) + ".pdf",
                        background=BackgroundTask(lambda: os.path.exists(path) and os.remove(path)))
