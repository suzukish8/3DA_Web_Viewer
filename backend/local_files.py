"""このアプリを動かしているパソコン上のファイルを、元のファイルと同じフォルダで扱うための API。

- ブラウザは、開いたファイルの場所(パス)を知らない。そこで、ファイルを開くダイアログをサーバー側(同じパソコン)で出し、
  パスを受け取る。パスが分かれば、.ant(注記の配置・表示の設定。Web 版は .ant、Windows 版は .antwin)を元のファイルと同じフォルダに読み書きできる。
- ドラッグ&ドロップなどで開いた(パスの分からない)ファイルは、これまでに開いたフォルダから同じ名前・大きさのファイルを探して、場所を割り出す。
- 安全のため、このパソコン自身(localhost)からの要求だけを受け付け、読めるのは STEP / QIF / .ant、書けるのは .ant だけ。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

router = APIRouter()

_STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recent_dirs.json")
_READABLE = (".stp", ".step", ".stpx", ".p21", ".qif", ".xml", ".ant")


def _local_only(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="このパソコンからの操作だけ受け付けます")


def _load_dirs() -> list[str]:
    try:
        with open(_STATE, encoding="utf-8") as f:
            d = json.load(f)
        return [x for x in d if isinstance(x, str)]
    except Exception:  # noqa: BLE001
        return []


def _remember_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    dirs = [x for x in _load_dirs() if os.path.normcase(x) != os.path.normcase(d)]
    dirs.insert(0, d)
    try:
        with open(_STATE, "w", encoding="utf-8") as f:
            json.dump(dirs[:30], f, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        pass


_DIALOG = r"""
import sys
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
p = filedialog.askopenfilename(
    title="Open",
    initialdir=sys.argv[1] if len(sys.argv) > 1 else "",
    filetypes=[("STEP / QIF", "*.stp *.step *.stpx *.p21 *.qif"), ("All files", "*.*")],
)
sys.stdout.write(p or "")
"""


@router.post("/api/open-dialog")
async def open_dialog(request: Request):
    """ファイルを開くダイアログ(このパソコンの画面に出る)。選んだファイルのパスを返す(取り消しなら空)。"""
    _local_only(request)
    dirs = _load_dirs()
    start = dirs[0] if dirs and os.path.isdir(dirs[0]) else ""
    try:
        import asyncio
        proc = await asyncio.to_thread(
            lambda: subprocess.run([sys.executable, "-c", _DIALOG, start], capture_output=True, text=True, timeout=600))
        path = (proc.stdout or "").strip()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"ダイアログを開けませんでした: {e}")
    if path:
        path = os.path.normpath(path)
        _remember_dir(path)
    return {"path": path}


@router.get("/api/file")
async def read_file(request: Request, path: str):
    _local_only(request)
    p = os.path.abspath(path)
    if not p.lower().endswith(_READABLE) or not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="ファイルがありません")
    return FileResponse(p, filename=os.path.basename(p))


@router.get("/api/locate")
async def locate(request: Request, name: str, size: int = -1):
    """これまでに開いたフォルダの中から、同じ名前(・大きさ)のファイルを探して、パスを返す。"""
    _local_only(request)
    for d in _load_dirs():
        p = os.path.join(d, os.path.basename(name))
        if os.path.isfile(p) and (size < 0 or os.path.getsize(p) == size):
            return {"path": p}
    return {"path": ""}


def _ant_path(path: str) -> str:
    base = os.path.splitext(os.path.abspath(path))[0]
    return base + ".ant"


@router.get("/api/ant")
async def read_ant(request: Request, path: str):
    _local_only(request)
    p = _ant_path(path)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="no ant")
    with open(p, encoding="utf-8") as f:
        return PlainTextResponse(f.read())


@router.put("/api/ant")
async def write_ant(request: Request, path: str):
    _local_only(request)
    p = _ant_path(path)
    body = (await request.body()).decode("utf-8")
    if not body.strip():
        raise HTTPException(status_code=400, detail="empty")
    # Web 版の設定は .ant、Windows 版(3DA B&M Viewer)の設定は .antwin。既存の .ant が Windows 版の形式なら、上書きする前に .antwin へ残す
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                head = f.read(4000)
            m = re.search(r"<\?xml[^>]*\?>\s*<([A-Za-z0-9_]+)", head) or re.search(r"<([A-Za-z0-9_]+)", head)
            legacy = os.path.splitext(p)[0] + ".antwin"
            if m and m.group(1) != "WebViewerAnnotationLayout" and not os.path.exists(legacy):
                shutil.copy2(p, legacy)
        except Exception:  # noqa: BLE001
            pass
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(body)
    os.replace(tmp, p)
    _remember_dir(path)
    return {"saved": p}
