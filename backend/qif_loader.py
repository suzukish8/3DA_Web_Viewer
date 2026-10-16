"""QIFファイルの読み込み。qif_viewer のパーサ/テッセレータを再利用し、
STEPと同じ形式のレスポンス(mesh + pmi)に変換する。"""
from __future__ import annotations

import os
import sys

_ROOT = os.environ.get("QIF_VIEWER_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qif_viewer.parsing.qif_document import load_qif_document  # noqa: E402
from qif_viewer.tessellate.mesh_builder import build_mesh  # noqa: E402
from qif_viewer.annotations import types_gdt  # noqa: E402,F401  (特性タイプの登録)
from qif_viewer.annotations.registry import get_spec  # noqa: E402
from qif_viewer.annotations.builder import _iso_value_text  # noqa: E402

TYPE_JA = {
    "Flatness": "平面度", "Straightness": "真直度", "Circularity": "真円度", "Cylindricity": "円筒度",
    "Parallelism": "平行度", "Perpendicularity": "直角度", "Angularity": "傾斜度",
    "Position": "位置度", "Concentricity": "同心度", "Symmetry": "対称度",
    "CircularRunout": "円周振れ", "TotalRunout": "全振れ",
    "LineProfile": "線の輪郭度", "SurfaceProfile": "面の輪郭度",
    "Width": "幅", "Diameter": "直径", "Radius": "半径", "Length": "長さ", "Distance": "距離",
    "Angle": "角度", "Thickness": "厚さ", "Depth": "深さ",
}


def load_qif(path: str) -> tuple[dict, dict]:
    doc = load_qif_document(path)
    mesh, cell_offsets = build_mesh(doc)
    if mesh.n_points == 0:
        raise ValueError("QIFから形状を生成できませんでした(B-Rep情報が見つかりません)")

    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (n_tri, 3)
    bounds = mesh.bounds
    mesh_json = {
        "positions": mesh.points.astype(float).ravel().tolist(),
        "indices": faces.astype(int).ravel().tolist(),
        "bbox": {"min": [bounds[0], bounds[2], bounds[4]], "max": [bounds[1], bounds[3], bounds[5]]},
        "faceCount": int(faces.shape[0]),
        "faceRanges": [[int(a), int(b)] for a, b, _fid in cell_offsets],
    }

    ranges_by_face: dict[str, list[list[int]]] = {}
    for start, end, face_id in cell_offsets:
        ranges_by_face.setdefault(face_id, []).append([start, end])

    def datum_labels(drf_id):
        drf = doc.datum_reference_frames.get(drf_id) if drf_id else None
        if drf is None:
            return []
        out = []
        for did, _prec in drf.entries:
            d = doc.datums.get(did)
            if d is not None:
                out.append(d.label)
        return out

    tolerances = []
    for nom in doc.characteristic_nominals.values():
        definition = doc.characteristic_definitions.get(nom.definition_id)
        face_ids: list[str] = []
        for fn_id in nom.feature_nominal_ids:
            fn = doc.feature_nominals.get(fn_id)
            if fn is not None:
                face_ids.extend(fn.entity_internal_ids)
        tri_ranges: list[list[int]] = []
        for fid in dict.fromkeys(face_ids):
            tri_ranges.extend(ranges_by_face.get(fid, []))
        spec = get_spec(nom.type_name)
        own_fns = set(nom.feature_nominal_ids)
        attached = [d.label for d in doc.datums.values() if own_fns & set(d.feature_nominal_ids)]
        try:
            value_text = _iso_value_text(definition, nom) if definition else "?"
        except Exception:
            value_text = "?"
        tolerances.append({
            "layer": nom.type_name,
            "symbol_kind": spec.symbol_kind,
            "dimension_icon": spec.dimension_icon,
            "designator": nom.designator or "",
            "value_text": value_text or "?",
            "datum_refs": datum_labels(definition.datum_reference_frame_id) if definition else [],
            "attached_datums": attached,
            "type": nom.type_name,
            "type_ja": TYPE_JA.get(nom.type_name, nom.type_name),
            "value_mm": definition.tolerance_value if definition else None,
            "name": nom.designator or nom.id,
            "datums": datum_labels(definition.datum_reference_frame_id) if definition else [],
            "face_triangle_ranges": tri_ranges,
        })

    datum_items = []
    for d in doc.datums.values():
        fids: list[str] = []
        for fn_id in d.feature_nominal_ids:
            fn = doc.feature_nominals.get(fn_id)
            if fn is not None:
                fids.extend(fn.entity_internal_ids)
        ranges: list[list[int]] = []
        for fid in dict.fromkeys(fids):
            ranges.extend(ranges_by_face.get(fid, []))
        datum_items.append({"letter": d.label, "names": [], "face_triangle_ranges": ranges})

    saved_views = []
    try:
        from qif_viewer.parsing.saved_views import read_saved_views
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = read_saved_views(fh.read())
        for name, st in raw.items():
            if isinstance(st, dict) and {"position", "focal_point", "view_up", "parallel_scale"} <= st.keys():
                saved_views.append({"name": name, "source": "qif", "position": st["position"],
                                    "focal_point": st["focal_point"], "up": st["view_up"],
                                    "parallel_scale": st["parallel_scale"]})
    except Exception:
        pass

    # 同じデータム形体にサイズ寸法と幾何公差の両方がある場合は、幾何公差の方だけに三角記号を書く
    geo_letters = {l for t in tolerances if t["symbol_kind"] != "dimension" for l in t["attached_datums"]}
    for t in tolerances:
        if t["symbol_kind"] == "dimension":
            t["attached_datums"] = [l for l in t["attached_datums"] if l not in geo_letters]

    pmi = {
        "geometric_tolerances": tolerances,
        "datums": datum_items,
        "tessellated": [],
        "saved_views": saved_views,
        "unit_factor_to_mm": 1.0,
    }
    return mesh_json, pmi
