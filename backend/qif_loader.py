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
from qif_viewer.annotations.builder import _iso_value_text, _total_feature_instances  # noqa: E402
from qif_viewer.parsing.xml_utils import id_list  # noqa: E402

TYPE_JA = {
    "Flatness": "平面度", "Straightness": "真直度", "Circularity": "真円度", "Cylindricity": "円筒度",
    "Parallelism": "平行度", "Perpendicularity": "直角度", "Angularity": "傾斜度",
    "Position": "位置度", "Concentricity": "同心度", "Symmetry": "対称度",
    "CircularRunout": "円周振れ", "TotalRunout": "全振れ",
    "LineProfile": "線の輪郭度", "SurfaceProfile": "面の輪郭度",
    "Width": "幅", "Diameter": "直径", "Radius": "半径", "Length": "長さ", "Distance": "距離",
    "Angle": "角度", "Thickness": "厚さ", "Depth": "深さ",
}


SURFACE_JA_QIF = {
    "PlaneSurface": "平面", "CylinderSurface": "円筒面", "ConeSurface": "円すい面",
    "NurbsSurface": "自由曲面(NURBS)", "UnknownSurface": "不明", "OcctSurface": "面(JT)",
}


def _qif_face_info(doc, face_id: str) -> dict:
    face = doc.faces.get(face_id)
    surface = doc.surfaces.get(face.surface_id) if face is not None else None
    kind = type(surface).__name__ if surface is not None else ""
    label = (face.label if face is not None else "") or ""
    if label.startswith("#"):          # 元ファイル内部の参照文字列は名前として出さない
        label = ""
    return {"id": face_id, "name": label, "surface": SURFACE_JA_QIF.get(kind, kind or "不明"),
            "surface_kw": kind}


def _qif_topology(doc) -> dict:
    """計測(Tool > Measurement)用の頂点・稜線(エッジ)・面ごとの稜線の一覧。ID は QIF の id 文字列。"""
    def vec(a):
        return [float(x) for x in a]

    vertices = [{"id": vid, "p": vec(v.point)} for vid, v in doc.vertices.items()]
    edges = []
    for eid, e in doc.edges.items():
        cv = doc.curves.get(e.curve_id)
        kind = type(cv).__name__ if cv is not None else ""
        info = {"type": {"LineCurve": "LINE", "ArcCurve": "CIRCLE", "EllipseCurve": "ELLIPSE", "NurbsCurve": "BSPLINE"}.get(kind, "OTHER"),
                "curve_id": e.curve_id}
        if kind == "ArcCurve":
            info.update({"center": vec(cv.center), "axis": vec(cv.normal), "radius": float(cv.radius)})
        v1, v2 = doc.vertices.get(e.vertex_beg_id), doc.vertices.get(e.vertex_end_id)
        edges.append({"id": eid, "v1": e.vertex_beg_id, "v2": e.vertex_end_id,
                      "p1": vec(v1.point) if v1 is not None else None, "p2": vec(v2.point) if v2 is not None else None, **info})
    face_edges: dict[str, list] = {}
    for fid, f in doc.faces.items():
        es = []
        for lid in f.loop_ids:
            lp = doc.loops.get(lid)
            if lp is not None:
                es.extend(ce.edge_id for ce in lp.coedges)
        if es:
            face_edges[str(fid)] = sorted(set(es))
    return {"vertices": vertices, "edges": edges, "face_edges": face_edges}


def _qif_attributes(path: str) -> list[dict]:
    """QIF の中にある属性情報(ファイル・ヘッダ・ユーザー定義属性・部品の項目)を、表示用にまとめる。"""
    import xml.etree.ElementTree as ET

    def ln(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []
    sections: list[dict] = []

    def user_attrs(el) -> list[dict]:
        out = []
        for a in el:
            if ln(a.tag) == "Attributes":
                for it in a:
                    if ln(it.tag).startswith("Attribute") and it.get("name") is not None and (it.get("value") or "") != "":
                        out.append({"name": it.get("name"), "value": it.get("value")})
        return out

    def leaves(el, prefix: str, depth: int, out: list, limit: int = 120) -> None:
        for ch in el:
            t = ln(ch.tag)
            if t in ("Attributes",) or t == "Id" or t.endswith("Ids") or t.endswith("IdSet") or len(out) >= limit:      # 内部の参照番号の並びは出さない
                continue
            txt = (ch.text or "").strip()
            name = (prefix + "/" + t) if prefix else t
            if len(ch) == 0:
                if txt:
                    out.append({"name": name, "value": txt})
            elif depth > 0 and t not in ("GeometrySet", "TopologySet", "FeatureNominalSet", "FeatureItemSet", "CharacteristicNominalSet", "CharacteristicItemSet", "CharacteristicDefinitionSet"):
                leaves(ch, name, depth - 1, out, limit)

    items = []
    if root.get("versionQIF"):
        items.append({"name": "QIF バージョン", "value": root.get("versionQIF")})
    for ch in root:
        if ln(ch.tag) == "QPId" and (ch.text or "").strip():
            items.append({"name": "QPId", "value": ch.text.strip()})
    for ch in root:
        if ln(ch.tag) == "Header":
            leaves(ch, "Header", 4, items)
    if items:
        sections.append({"title": "ファイル情報", "items": items})
    ua = user_attrs(root)
    if ua:
        sections.append({"title": "ユーザー定義属性", "items": ua})
    for prod in root:
        if ln(prod.tag) != "Product":
            continue
        for ch in prod:
            if ln(ch.tag) in ("PartSet", "AssemblySet"):
                for part in ch:
                    its: list[dict] = []
                    leaves(part, "", 3, its)
                    its.extend(user_attrs(part))
                    if its:
                        nm = next((i["value"] for i in its if i["name"] in ("Name", "PartName")), ln(part.tag))
                        sections.append({"title": "部品: " + nm, "items": its})
    return sections


def _qif_saved_views(path: str) -> list[dict]:
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

    return saved_views


def doc_to_response(doc, points, faces, cell_offsets, saved_views, attributes) -> tuple[dict, dict]:
    """QifDocument(QIF / JT 由来)とテッセレーション結果を、ブラウザ用の mesh + pmi に変換する。
    points: (n,3) 座標、faces: (n_tri,3) 三角形の頂点番号。"""
    bounds = (points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max())
    mesh_json = {
        "positions": points.astype(float).ravel().tolist(),
        "indices": faces.astype(int).ravel().tolist(),
        "bbox": {"min": [bounds[0], bounds[2], bounds[4]], "max": [bounds[1], bounds[3], bounds[5]]},
        "faceCount": int(faces.shape[0]),
        "faceRanges": [[int(a), int(b)] for a, b, _fid in cell_offsets],
        "faceInfo": [_qif_face_info(doc, fid) for _a, _b, fid in cell_offsets],
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

    def entity_ids(fn_id, seen=None):
        # Group 形体(パターン)は、含まれる形体の面をまとめて返す。同じ穴群を指す直径寸法と位置度が同じ塊になる
        fn = doc.feature_nominals.get(fn_id)
        if fn is None:
            return []
        ids = list(fn.entity_internal_ids)
        if getattr(fn, "kind", None) == "Group":
            seen = seen if seen is not None else set()
            if fn_id in seen:
                return ids
            seen.add(fn_id)
            for sub in id_list(fn.raw_element, "FeatureNominalIds"):
                ids.extend(entity_ids(sub, seen))
        return ids

    def first_instance_ids(fn_id, seen=None):
        # パターン(Group や複数の形体)は、重心ではなく最初の 1 個(実在する穴など)を指す
        fn = doc.feature_nominals.get(fn_id)
        if fn is None:
            return []
        if fn.entity_internal_ids:
            return list(fn.entity_internal_ids)
        if getattr(fn, "kind", None) == "Group":
            seen = seen if seen is not None else set()
            if fn_id in seen:
                return []
            seen.add(fn_id)
            for sub in id_list(fn.raw_element, "FeatureNominalIds"):
                ids = first_instance_ids(sub, seen)
                if ids:
                    return ids
        return []

    tolerances = []
    for nom in doc.characteristic_nominals.values():
        definition = doc.characteristic_definitions.get(nom.definition_id)
        face_ids: list[str] = []
        for fn_id in nom.feature_nominal_ids:
            face_ids.extend(entity_ids(fn_id))
        tri_ranges: list[list[int]] = []
        for fid in dict.fromkeys(face_ids):
            tri_ranges.extend(ranges_by_face.get(fid, []))
        # 複数のサイズ形体(穴など)を 1 つの寸法で指すときの「N×」(Diameter/Radius のみ)
        count = 1
        if nom.type_name in ("Diameter", "Radius"):
            try:
                count = max(1, _total_feature_instances(list(nom.feature_nominal_ids), doc))
            except Exception:
                count = 1
        fns = [doc.feature_nominals.get(i) for i in nom.feature_nominal_ids]
        is_pattern = any(getattr(f, "kind", None) == "Group" for f in fns) or (
            len(fns) > 1 and all(f is not None and type(f).__name__ == "CylinderFeatureNominal" for f in fns))
        anchor_ranges: list[list[int]] = []
        for fn_id in nom.feature_nominal_ids:
            ids = first_instance_ids(fn_id)
            if ids:
                for fid in dict.fromkeys(ids):
                    anchor_ranges.extend(ranges_by_face.get(fid, []))
                break
        spec = get_spec(nom.type_name)
        own_fns = set(nom.feature_nominal_ids)
        attached = [d.label for d in doc.datums.values() if own_fns & set(d.feature_nominal_ids)]
        try:
            value_text = _iso_value_text(definition, nom) if definition else "?"
        except Exception:
            value_text = "?"
        tolerances.append({
            "id": nom.id,                      # QIF の CharacteristicNominal の id(Source パネルとの相互ハイライトに使う)
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
            "anchor_triangle_ranges": anchor_ranges,
            "pattern": is_pattern,
            "_fns": list(nom.feature_nominal_ids),
            "count": count,
        })

    # 幾何公差がグループ(パターン)でまとめて指す穴に、同じ直径寸法が 1 つずつ付いているとき:
    # 1 つだけ残して「N×」を付け、残りの寸法は書かない
    drop: set[int] = set()
    for t in tolerances:
        if t["symbol_kind"] == "dimension" or len(t["_fns"]) != 1:
            continue
        g = doc.feature_nominals.get(t["_fns"][0])
        if getattr(g, "kind", None) != "Group":
            continue
        subs = set(id_list(g.raw_element, "FeatureNominalIds"))
        dims = [i for i, d in enumerate(tolerances)
                if d["layer"] in ("Diameter", "Radius") and len(d["_fns"]) == 1 and d["_fns"][0] in subs and i not in drop]
        if len(dims) >= 2 and len({tolerances[i]["layer"] + tolerances[i]["value_text"] for i in dims}) == 1:
            tolerances[dims[0]]["count"] = sum(tolerances[i]["count"] for i in dims)
            keep = tolerances[dims[0]]
            for i in dims[1:]:                      # ハイライトは残した寸法で全部の穴を光らせる
                keep["face_triangle_ranges"] = keep["face_triangle_ranges"] + tolerances[i]["face_triangle_ranges"]
            keep["pattern"] = True                 # 塊のまとめ・アンカー・寸法線は最初の 1 個の穴で行う
            drop.update(dims[1:])
    tolerances = [t for i, t in enumerate(tolerances) if i not in drop]
    for t in tolerances:
        t.pop("_fns", None)

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

    # 同じデータム形体にサイズ寸法と幾何公差の両方がある場合は、幾何公差の方だけに三角記号を書く
    geo_letters = {l for t in tolerances if t["symbol_kind"] != "dimension" for l in t["attached_datums"]}
    for t in tolerances:
        if t["symbol_kind"] == "dimension":
            t["attached_datums"] = [l for l in t["attached_datums"] if l not in geo_letters]

    pmi = {
        "geometric_tolerances": tolerances,
        "topology": _qif_topology(doc),
        "datums": datum_items,
        "tessellated": [],
        "saved_views": saved_views,
        "attributes": attributes,
        "unit_factor_to_mm": 1.0,
    }
    return mesh_json, pmi


def load_qif(path: str) -> tuple[dict, dict]:
    doc = load_qif_document(path)
    mesh, cell_offsets = build_mesh(doc)
    if mesh.n_points == 0:
        raise ValueError("QIFから形状を生成できませんでした(B-Rep情報が見つかりません)")
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (n_tri, 3)
    return doc_to_response(doc, mesh.points, faces, cell_offsets, _qif_saved_views(path), _qif_attributes(path))


def _jt_attributes(path: str) -> list[dict]:
    from qif_viewer.jt.reader import SEGMENT_NAMES, read_jt

    jt = read_jt(path)
    counts: dict[str, int] = {}
    for s in jt.segments:
        n = SEGMENT_NAMES.get(s.type, f"type {s.type}")
        counts[n] = counts.get(n, 0) + 1
    return [{"title": "ファイル情報", "items": [{"name": "JT バージョン", "value": jt.version}] + [
        {"name": "セグメント: " + n, "value": str(c)} for n, c in sorted(counts.items())]}]


def load_jt(path: str) -> tuple[dict, dict]:
    """JT ファイルを読む。形状は JT 自身の三角形データ(Shape LOD)から、PMI は JT のメタデータから得る。"""
    from qif_viewer.jt.loader import PointsFaces, apply_jt_pmi, build_jt_geometry
    from qif_viewer.jt.report import jt_report_text

    geom = build_jt_geometry(path)
    apply_jt_pmi(geom, PointsFaces(geom.points, geom.triangles), path)
    mesh, pmi = doc_to_response(geom.doc, geom.points, geom.triangles, geom.cell_offsets, [], _jt_attributes(path))
    pmi["source_text"] = jt_report_text(path, geom.doc)
    try:
        pmi["tessellated"] = _jt_graphic_groups(path, geom.doc, pmi)
    except Exception:  # noqa: BLE001 - drawn PMI is optional
        pmi["tessellated"] = []
    return mesh, pmi


def _jt_graphic_groups(path: str, doc, pmi: dict) -> list[dict]:
    """JT に描かれている PMI(記入枠・引出線・寸法線・文字の線分)を、STEP の tessellated と同じ形に直す。
    グループ名は、対応する特性の name(= "jt-<エンティティ番号>")に合わせ、選択・ハイライトを連動させる。"""
    from qif_viewer.jt.graphic import extract_graphic
    from qif_viewer.jt.pmi import KIND_DATUM, extract_pmi
    from qif_viewer.jt.reader import read_jt

    jt = read_jt(path)
    graphic = extract_graphic(jt)
    raw = {(e.segment_offset, e.run_start): e for e in extract_pmi(jt).entities}
    index = getattr(doc, "step_pmi_name_index", None) or {}
    tol_by_id = {t["id"]: t for t in pmi["geometric_tolerances"]}
    groups: list[dict] = []
    for key, g in sorted(graphic.items()):
        lines = [[[float(x) for x in p] for p in pl.points] for pl in g.polylines if len(pl.points) >= 2]
        if not lines:
            continue
        e = raw.get(key)
        name = f"jt-gfx-{key[0]}-{key[1]}"
        if e is not None:
            name = f"jt-{e.id}"
            for aid in index.get(name, []):
                t = tol_by_id.get(aid[5:] if aid.startswith("char-") else "")
                if t is not None:
                    t["name"] = name          # every row of a stacked frame shares the group
            if e.kind == KIND_DATUM:
                for d in pmi["datums"]:
                    if d["letter"] == e.props.get("label"):
                        d["names"].append(name)
        groups.append({"name": name, "polylines": lines})
    return groups
