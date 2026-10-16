"""ブラウザ版(Pyodide)用スタブ。mapbox_earcut の代わりに JS 版 earcut (wasm_loader.js が読み込む) を呼ぶ。
Pyodide 以外では ImportError になり、qif_viewer 側の純 Python 版にフォールバックする。"""
import numpy as np
import js
from pyodide.ffi import to_js

if not hasattr(js, "earcut"):
    raise ImportError("earcut (JS) is not loaded")


def triangulate_float64(verts2d, ring_ends):
    flat = to_js(np.asarray(verts2d, dtype=np.float64).ravel().tolist())
    holes = to_js([int(e) for e in np.asarray(ring_ends)[:-1]])
    out = js.earcut(flat, holes, 2)
    return np.asarray(out.to_py(), dtype=np.uint32)
