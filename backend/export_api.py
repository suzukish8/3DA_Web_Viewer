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
    solids = [
        DxfSolid(layer=str(s.get("layer", "0")), rgb=_rgb(s.get("rgb")), points=np.asarray(s["points"], dtype=float).reshape(-1, 2))
        for s in body.get("solids", []) if len(s.get("points", [])) >= 3
    ]
    labels = [
        DxfLabel(layer=str(t.get("layer", "0")), rgb=_rgb(t.get("rgb")), position=(float(t["p"][0]), float(t["p"][1])),
                 text=str(t["text"]), height=float(t["h"]))
        for t in body.get("labels", [])
    ]
    widths = body.get("line_widths") or [0.5, 0.25]
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
