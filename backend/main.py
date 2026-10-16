"""
3DA Web Viewer - バックエンド (FastAPI)

- POST /api/upload : STEPファイルを受け取り、
    1) OCCT(OCP)で形状をテッセレーション(三角形メッシュ化)
    2) step_pmi.py で意味的PMI(幾何公差・データム)を抽出
  してJSONで返す。
- それ以外のパスは frontend/ 以下の静的ファイル(three.jsビューア)を配信する。
"""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

from step_pmi import extract_pmi

app = FastAPI(title="3DA Web Viewer API")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB


def tessellate(shape) -> tuple[dict, list[dict]]:
    """形状をテッセレーションし、three.js の BufferGeometry 用データを作る。
    あわせて、各OCCT面ごとの(三角形インデックス範囲, 境界頂点座標の集合)も返す。
    STEP側のエンティティIDとOCCTの対応表(XSControl_TransferReader)がこの環境では
    信頼できなかった(Model().Number()が常に0、ShapeResult()はNull形状になりがち)ため、
    PMIとの対応付けは、この境界頂点座標を使った幾何マッチングで行う
    (see: step_pmi.extract_face_anchor_points, upload_step内のマッチング処理)。"""
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    # 形状サイズに応じた許容誤差(粗すぎず細かすぎない値)
    deflection = max(diag * 0.0015, 1e-3)

    BRepMesh_IncrementalMesh(shape, deflection)

    positions: list[float] = []
    indices: list[int] = []
    vertex_offset = 0
    face_records: list[dict] = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tri_start = len(indices) // 3
            trsf = loc.Transformation()
            is_reversed = face.Orientation() == TopAbs_REVERSED
            n_nodes = tri.NbNodes()
            node_set = set()
            for i in range(1, n_nodes + 1):
                p = tri.Node(i).Transformed(trsf)
                positions.extend((p.X(), p.Y(), p.Z()))
                node_set.add((round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6)))
            n_tris = tri.NbTriangles()
            for i in range(1, n_tris + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                if is_reversed:
                    n1, n2, n3 = n3, n2, n1
                indices.extend((
                    vertex_offset + n1 - 1,
                    vertex_offset + n2 - 1,
                    vertex_offset + n3 - 1,
                ))
            vertex_offset += n_nodes

            face_records.append({
                "tri_range": [tri_start, len(indices) // 3],
                "node_set": node_set,
            })
        exp.Next()

    mesh = {
        "positions": positions,
        "indices": indices,
        "bbox": {"min": [xmin, ymin, zmin], "max": [xmax, ymax, zmax]},
        "faceCount": vertex_offset and len(indices) // 3,
    }
    return mesh, face_records


def match_face_triangle_ranges(anchor_points: list[list[float]], face_records: list[dict]) -> list[list[int]]:
    """PMIの対象面の境界頂点座標(STEP側)を、テッセレーション済みメッシュの各面の
    頂点集合と突き合わせ、一致した面の三角形インデックス範囲を返す。"""
    if not anchor_points:
        return []
    rounded = {(round(p[0], 6), round(p[1], 6), round(p[2], 6)) for p in anchor_points}

    scored = []
    for rec in face_records:
        overlap = len(rounded & rec["node_set"])
        if overlap > 0:
            scored.append((overlap, rec["tri_range"]))
    if not scored:
        return []
    best = max(s[0] for s in scored)
    # ほぼ同点(頂点を全部/ほぼ含む)の面をすべて採用する(1つのPMIが複数面にまたがる場合に対応)
    threshold = max(1, best - 1) if best > 2 else 1
    return [tri_range for overlap, tri_range in scored if overlap >= threshold]


@app.post("/api/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.stp"
    suffix = os.path.splitext(filename)[1] or ".stp"

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ファイルが大きすぎます(200MB上限)")
    if not data:
        raise HTTPException(status_code=400, detail="空のファイルです")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if suffix.lower() in (".qif", ".xml"):
            from qif_loader import load_qif
            try:
                mesh, pmi = load_qif(tmp_path)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"QIFファイルとして読み込めませんでした: {e}")
            return JSONResponse({"filename": filename, "mesh": mesh, "pmi": pmi})
        reader = STEPControl_Reader()
        status = reader.ReadFile(tmp_path)
        if status != IFSelect_RetDone:
            raise HTTPException(status_code=400, detail="STEPファイルとして読み込めませんでした")
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape is None or shape.IsNull():
            raise HTTPException(status_code=400, detail="有効な形状が見つかりませんでした")

        mesh, face_records = tessellate(shape)
        mesh["faceRanges"] = [r["tri_range"] for r in face_records]

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")
        pmi = extract_pmi(text)

        # 幾何公差ごとに、対象面の三角形インデックス範囲(ハイライト表示用)を
        # 境界頂点座標のマッチングで求めて付与する
        for tol in pmi.get("geometric_tolerances", []):
            tol["face_triangle_ranges"] = match_face_triangle_ranges(tol.pop("anchor_points", []), face_records)

        # メッシュの各面(OCC の面)に、対応する STEP の面の ID・名前・曲面の種類を付ける(境界頂点の一致で対応付け)
        point_index: dict = {}
        for ri, rec in enumerate(face_records):
            for pt in rec["node_set"]:
                point_index.setdefault(pt, []).append(ri)
        face_info = [None] * len(face_records)
        # 候補(STEP の面, メッシュの面)ごとに「境界頂点が一致した数」を数え、一致率の高い組から一対一で確定する
        cands = []
        infos = pmi.pop("face_infos", [])
        for fi_idx, fi in enumerate(infos):
            votes: dict = {}
            n = max(len(fi["anchor_points"]), 1)
            for p in fi["anchor_points"]:
                for ri in point_index.get((round(p[0], 6), round(p[1], 6), round(p[2], 6)), ()):
                    votes[ri] = votes.get(ri, 0) + 1
            for ri, v in votes.items():
                cands.append((v / n, v, fi_idx, ri))
        cands.sort(key=lambda c: (-c[0], -c[1]))
        used_step: set = set()
        for frac, v, fi_idx, ri in cands:
            if fi_idx in used_step or face_info[ri] is not None:
                continue
            if v < min(2, max(len(infos[fi_idx]["anchor_points"]), 1)):
                continue
            fi = infos[fi_idx]
            face_info[ri] = {"id": fi["id"], "name": fi["name"], "surface": fi["surface"], "surface_kw": fi["surface_kw"]}
            used_step.add(fi_idx)
        mesh["faceInfo"] = face_info

        for a in pmi.get("annotations", []):
            a["face_triangle_ranges"] = match_face_triangle_ranges(a.pop("anchor_points", []), face_records)
        for d in pmi.get("datums", []):
            d["face_triangle_ranges"] = match_face_triangle_ranges(d.pop("anchor_points", []), face_records)
        from pmi_display import decorate_step_tolerances
        decorate_step_tolerances(pmi.get("geometric_tolerances", []))

        return JSONResponse({
            "filename": filename,
            "mesh": mesh,
            "pmi": pmi,
        })
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


from export_api import router as export_router  # noqa: E402
app.include_router(export_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
