"""ブラウザ版(Pyodide)用に、qif_viewer のパーサ/テッセレータを frontend/py/qif_viewer へコピーする。
qif_viewer 側を更新したら再実行し、frontend/py/ をコミットする。
使い方: python sync_qif_web.py [qif_viewer のルート(既定: ..)]"""
import os, shutil, sys, json

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, ".."))
src = os.path.join(root, "qif_viewer")
dst = os.path.join(here, "frontend", "py", "qif_viewer")

SKIP = {"edge_lines.py", "vertex_points.py"}          # pyvista を使う(メッシュ表示には不要)
PKGS = ["model", "parsing", "tessellate", "tessellate/surfaces", "annotations"]

if os.path.isdir(dst):
    shutil.rmtree(dst)
files = []
for pkg in [""] + PKGS:
    d = os.path.join(dst, pkg); os.makedirs(d, exist_ok=True)
    for f in sorted(os.listdir(os.path.join(src, pkg))):
        if f.endswith(".py") and f not in SKIP:
            if pkg == "" and f != "__init__.py":
                continue
            shutil.copy(os.path.join(src, pkg, f), os.path.join(d, f))
            files.append(((pkg + "/" if pkg else "") + f))
with open(os.path.join(dst, "manifest.json"), "w") as fh:
    json.dump(files, fh)
print(len(files), "files ->", dst)
