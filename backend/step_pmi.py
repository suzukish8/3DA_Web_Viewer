"""
STEP AP242 の生テキストを直接パースして、意味的PMI（GD&T: 幾何公差・データム・寸法）を抽出する。

OCCTのXCAFDoc_DimTolTool経由の高レベルAPIは、このプロジェクトで使用している
cadquery-ocp(OCCT 7.9系)ビルドでは複合エンティティ(complex entity)の値解決が
うまく働かないケースがあったため、STEP(ISO 10303-21)のテキスト構造を
直接パースする軽量な自前パーサーで代替する。

STEPのDATA部は次の2パターンのレコードから成る:
  単純エンティティ: #123=KEYWORD(arg1,arg2,...);
  複合エンティティ: #123=(
                        KEYWORD_A(args...)
                        KEYWORD_B(args...)
                      );
本パーサーはどちらのパターンも「#id -> [(KEYWORD, [フィールド...]), ...]」という
共通の形に正規化して扱う。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# STEPのGD&T記号種別 -> 日本語表記（GD&T用語は「形体」を用いる。「特徴」は使わない）
TOLERANCE_TYPE_JA = {
    "FLATNESS_TOLERANCE": "平面度",
    "STRAIGHTNESS_TOLERANCE": "真直度",
    "ROUNDNESS_TOLERANCE": "真円度",
    "CIRCULARITY_TOLERANCE": "真円度",
    "CYLINDRICITY_TOLERANCE": "円筒度",
    "PARALLELISM_TOLERANCE": "平行度",
    "PERPENDICULARITY_TOLERANCE": "直角度",
    "ANGULARITY_TOLERANCE": "傾斜度",
    "POSITION_TOLERANCE": "位置度",
    "CONCENTRICITY_TOLERANCE": "同心度/同軸度",
    "COAXIALITY_TOLERANCE": "同軸度",
    "SYMMETRY_TOLERANCE": "対称度",
    "CIRCULAR_RUNOUT_TOLERANCE": "円周振れ",
    "TOTAL_RUNOUT_TOLERANCE": "全振れ",
    "SURFACE_PROFILE_TOLERANCE": "面の輪郭度",
    "PROFILE_OF_SURFACE_TOLERANCE": "面の輪郭度",
    "LINE_PROFILE_TOLERANCE": "線の輪郭度",
    "PROFILE_OF_LINE_TOLERANCE": "線の輪郭度",
}

_MEASURE_KEYWORDS = (
    "POSITIVE_LENGTH_MEASURE",
    "LENGTH_MEASURE",
    "PLANE_ANGLE_MEASURE",
    "POSITIVE_PLANE_ANGLE_MEASURE",
    "COUNT_MEASURE",
)

MODIFIER_SYMBOLS = {
    "MAXIMUM_MATERIAL_REQUIREMENT": "Ⓜ",   # Ⓜ
    "LEAST_MATERIAL_REQUIREMENT": "Ⓛ",     # Ⓛ
    "FREE_STATE": "Ⓕ",                     # Ⓕ
    "TANGENT_PLANE": "Ⓣ",                  # Ⓣ
    "PROJECTED_TOLERANCE_ZONE": "Ⓟ",       # Ⓟ
}


def _modifier_symbols(text: str) -> str:
    """".MAXIMUM_MATERIAL_REQUIREMENT." 等の列挙値から、◯付き記号を並べて返す。"""
    out = ""
    for name in re.findall(r"\.([A-Z_]+)\.", text or ""):
        sym = MODIFIER_SYMBOLS.get(name)
        if sym and sym not in out:
            out += sym
    return out


_TOKEN_RE = re.compile(r"([A-Z][A-Z0-9_]*)\s*\(")


@dataclass
class Entity:
    id: int
    # 1つの単純エンティティなら要素1つ、複合エンティティなら複数
    parts: list[tuple[str, str]] = field(default_factory=list)  # (KEYWORD, 生の引数文字列)


def _find_matching_paren(text: str, open_idx: int) -> int:
    """text[open_idx] == '(' として、対応する ')' のインデックスを返す(文字列リテラル内は無視)。"""
    depth = 0
    in_str = False
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "'":
                # STEPの文字列内 '' はエスケープされた ' 一文字
                if i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                in_str = False
        else:
            if c == "'":
                in_str = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _split_top_level_args(args: str) -> list[str]:
    """カンマ区切りの引数文字列を、括弧・文字列の深さを見て最上位だけで分割する。"""
    fields: list[str] = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    n = len(args)
    while i < n:
        c = args[i]
        if in_str:
            if c == "'":
                if i + 1 < n and args[i + 1] == "'":
                    i += 2
                    continue
                in_str = False
        else:
            if c == "'":
                in_str = True
            elif c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif c == "," and depth == 0:
                fields.append(args[start:i].strip())
                start = i + 1
        i += 1
    tail = args[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def parse_entities(step_text: str) -> dict[int, Entity]:
    """DATA;...ENDSEC; を走査し、id -> Entity の辞書を返す。"""
    m = re.search(r"\bDATA\s*;", step_text)
    data_text = step_text[m.end():] if m else step_text

    entities: dict[int, Entity] = {}
    # レコード先頭: 行頭が # で始まる "#123="
    for rec_m in re.finditer(r"#(\d+)\s*=", data_text):
        eid = int(rec_m.group(1))
        pos = rec_m.end()
        # 直後の非空白文字を見る
        j = pos
        while j < len(data_text) and data_text[j].isspace():
            j += 1
        if j >= len(data_text):
            continue

        ent = Entity(id=eid)
        if data_text[j] == "(":
            # 複合エンティティ: 外側の括弧の中に KEYWORD(args) が連続する
            close = _find_matching_paren(data_text, j)
            if close == -1:
                continue
            inner = data_text[j + 1:close]
            for tok_m in _TOKEN_RE.finditer(inner):
                kw = tok_m.group(1)
                open_idx = tok_m.end() - 1
                kw_close = _find_matching_paren(inner, open_idx)
                if kw_close == -1:
                    continue
                ent.parts.append((kw, inner[open_idx + 1:kw_close]))
        else:
            # 単純エンティティ: KEYWORD(args);
            tok_m = _TOKEN_RE.match(data_text, j)
            if not tok_m:
                continue
            kw = tok_m.group(1)
            open_idx = tok_m.end() - 1
            kw_close = _find_matching_paren(data_text, open_idx)
            if kw_close == -1:
                continue
            ent.parts.append((kw, data_text[open_idx + 1:kw_close]))

        entities[eid] = ent
    return entities


def _ref(field_text: str) -> Optional[int]:
    field_text = field_text.strip()
    m = re.match(r"#(\d+)$", field_text)
    return int(m.group(1)) if m else None


def _refs_in(field_text: str) -> list[int]:
    return [int(x) for x in re.findall(r"#(\d+)", field_text)]


def decode_step_string(text: str) -> str:
    r"""STEP(ISO 10303-21)の文字列エスケープを展開する。
    \X2\<UTF-16の16進>\X0\ / \X4\<UTF-32の16進>\X0\ / \X\HH (ISO 8859-1の1バイト) / \S\c / \\ を処理する。"""
    BS = "\\"
    if BS not in text:
        return text
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch != BS:
            out.append(ch)
            i += 1
            continue
        if text.startswith(BS + "X2" + BS, i):
            j = text.find(BS + "X0" + BS, i + 4)
            if j != -1:
                hexs = text[i + 4:j]
                units = []
                for k in range(0, len(hexs) - len(hexs) % 4, 4):
                    try:
                        units.append(int(hexs[k:k + 4], 16))
                    except ValueError:
                        pass
                out.append(b"".join(u.to_bytes(2, "big") for u in units).decode("utf-16-be", errors="replace"))
                i = j + 4
                continue
        if text.startswith(BS + "X4" + BS, i):
            j = text.find(BS + "X0" + BS, i + 4)
            if j != -1:
                hexs = text[i + 4:j]
                for k in range(0, len(hexs) - len(hexs) % 8, 8):
                    try:
                        out.append(chr(int(hexs[k:k + 8], 16)))
                    except ValueError:
                        pass
                i = j + 4
                continue
        if text.startswith(BS + "X" + BS, i) and i + 5 <= n:
            try:
                out.append(chr(int(text[i + 3:i + 5], 16)))
                i += 5
                continue
            except ValueError:
                pass
        if text.startswith(BS + "S" + BS, i) and i + 4 <= n:
            out.append(chr(ord(text[i + 3]) + 128))
            i += 4
            continue
        if text.startswith(BS + BS, i):
            out.append(BS)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _unquote(field_text: str) -> str:
    field_text = field_text.strip()
    if field_text.startswith("'") and field_text.endswith("'"):
        return decode_step_string(field_text[1:-1].replace("''", "'"))
    return field_text


# SI接頭辞 -> 倍率 (STEPのenum値、ドットは剥がした形で比較する)
_SI_PREFIX_MULTIPLIER = {
    "EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9, "MEGA": 1e6, "KILO": 1e3,
    "HECTO": 1e2, "DECA": 1e1, "DECI": 1e-1, "CENTI": 1e-2, "MILLI": 1e-3,
    "MICRO": 1e-6, "NANO": 1e-9, "PICO": 1e-12, "FEMTO": 1e-15, "ATTO": 1e-18,
}


def _resolve_length_unit_factor(entities: dict[int, Entity], ref_id: Optional[int], _depth: int = 0) -> Optional[float]:
    """LENGTH_UNIT系エンティティを再帰的に辿り、ミリメートルへの換算係数を求める。"""
    if ref_id is None or ref_id not in entities or _depth > 6:
        return None
    ent = entities[ref_id]
    for kw, args in ent.parts:
        if kw == "SI_UNIT":
            fields = _split_top_level_args(args)
            if len(fields) < 2:
                continue
            prefix_raw = fields[0].strip()
            prefix = prefix_raw.strip(".").upper() if prefix_raw not in ("$", "") else None
            mult = _SI_PREFIX_MULTIPLIER.get(prefix, 1.0)
            return mult * 1000.0  # メートル -> ミリメートル
        if kw == "CONVERSION_BASED_UNIT":
            fields = _split_top_level_args(args)
            if len(fields) < 2:
                continue
            base_ref = _ref(fields[1])
            if base_ref is None or base_ref not in entities:
                continue
            for bkw, bargs in entities[base_ref].parts:
                if bkw not in ("LENGTH_MEASURE_WITH_UNIT", "MEASURE_WITH_UNIT"):
                    continue
                bfields = _split_top_level_args(bargs)
                if len(bfields) < 2:
                    continue
                m = re.match(r"[A-Z_]+\(([^()]*)\)", bfields[0].strip())
                if not m:
                    continue
                try:
                    value = float(m.group(1))
                except ValueError:
                    continue
                sub_factor = _resolve_length_unit_factor(entities, _ref(bfields[1]), _depth + 1)
                if sub_factor is not None:
                    return value * sub_factor
    return None


def find_length_unit_to_mm_factor(entities: dict[int, Entity]) -> float:
    """ファイル全体の長さの単位(GLOBAL_UNIT_ASSIGNED_CONTEXT)から、ミリメートルへの
    換算係数を求める。見つからない場合は1.0(ミリメートル前提)を返す。

    形状のB-rep座標や幾何公差の数値はこのグローバル単位系に従うため、
    インチ等で書かれたファイルではOCCT側(常にmm換算)とスケールを合わせる必要がある。
    (注: テッセレーションPMIの描画座標は、実測で確認した限りこの換算を適用しない方が
    実際のモデルと整合したため、あえて対象外にしている。CADシステムのPMI描画出力が
    グローバル単位系に従わない場合があるためと見られる。)"""
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw != "GLOBAL_UNIT_ASSIGNED_CONTEXT":
                continue
            fields = _split_top_level_args(args)
            if not fields:
                continue
            for ref in _refs_in(fields[0]):
                if ref not in entities:
                    continue
                kinds = {k for k, _ in entities[ref].parts}
                if "LENGTH_UNIT" not in kinds:
                    continue
                factor = _resolve_length_unit_factor(entities, ref)
                if factor is not None:
                    return factor
    return 1.0


def resolve_measure(entities: dict[int, Entity], ref_id: Optional[int]) -> Optional[float]:
    """磁石参照(#id)を辿り、MEASURE_WITH_UNIT系エンティティから数値を取り出す。"""
    if ref_id is None or ref_id not in entities:
        return None
    ent = entities[ref_id]
    for kw, args in ent.parts:
        # MEASURE_WITH_UNIT / LENGTH_MEASURE_WITH_UNIT / PLANE_ANGLE_MEASURE_WITH_UNIT などの派生も対象
        if kw == "MEASURE_WITH_UNIT" or kw.endswith("_MEASURE_WITH_UNIT"):
            fields = _split_top_level_args(args)
            if fields:
                inner = fields[0]
                mm = re.match(r"([A-Z_]+)\(([^()]*)\)", inner)
                if mm and mm.group(1) in _MEASURE_KEYWORDS:
                    try:
                        return float(mm.group(2))
                    except ValueError:
                        return None
    return None


def resolve_datum_letters(entities: dict[int, Entity], ref_id: Optional[int], _depth: int = 0) -> list[str]:
    """DATUM_SYSTEM / DATUM_REFERENCE_COMPARTMENT / DATUM_REFERENCE / DATUM を再帰的に辿り、データム文字の並びを得る。"""
    if ref_id is None or ref_id not in entities or _depth > 6:
        return []
    ent = entities[ref_id]
    letters: list[str] = []
    for kw, args in ent.parts:
        fields = _split_top_level_args(args)
        if kw == "DATUM":
            if fields:
                letter = _unquote(fields[-1])
                if letter:
                    letters.append(letter)
        elif kw in ("DATUM_SYSTEM",):
            for f in fields:
                for sub in _refs_in(f):
                    letters.extend(resolve_datum_letters(entities, sub, _depth + 1))
        elif kw in ("DATUM_REFERENCE_COMPARTMENT", "DATUM_REFERENCE", "DATUM_REFERENCE_ELEMENT"):
            found: list[str] = []
            for idx, f in enumerate(fields):
                if kw in ("DATUM_REFERENCE_COMPARTMENT", "DATUM_REFERENCE_ELEMENT") and idx == 5:
                    continue  # 末尾はモディファイア列
                ftxt = f.strip()
                if ftxt.startswith("COMMON_DATUM_LIST"):
                    # 共通データム(例: C-C): 要素ごとの文字をハイフンで連結する
                    parts: list[str] = []
                    for sub in _refs_in(ftxt):
                        parts.extend(resolve_datum_letters(entities, sub, _depth + 1))
                    if parts:
                        found.append("-".join(parts))
                    continue
                sub = _ref(f)
                if sub is not None:
                    found.extend(resolve_datum_letters(entities, sub, _depth + 1))
            if kw in ("DATUM_REFERENCE_COMPARTMENT", "DATUM_REFERENCE_ELEMENT") and len(fields) > 5 and found:
                mods = _modifier_symbols(fields[5])
                if mods:
                    found[-1] += mods   # 例: "A" -> "AⓂ"
            letters.extend(found)
    return letters


def _build_shape_aspect_relationships(entities: dict[int, Entity]) -> dict[int, list[int]]:
    """SHAPE_ASPECT_RELATIONSHIP(name, description, relating_shape_aspect, related_shape_aspect)
    から、複合形体(COMPOSITE_GROUP_SHAPE_ASPECT等)を構成する個々の形体への対応表を作る。"""
    rel: dict[int, list[int]] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw != "SHAPE_ASPECT_RELATIONSHIP":
                continue
            fields = _split_top_level_args(args)
            if len(fields) < 4:
                continue
            relating = _ref(fields[2])
            related = _ref(fields[3])
            if relating is not None and related is not None:
                rel.setdefault(relating, []).append(related)
    return rel


def _build_geometric_item_usage_map(entities: dict[int, Entity]) -> dict[int, list[int]]:
    """GEOMETRIC_ITEM_SPECIFIC_USAGE(name, description, definition, used_representation, identified_item)
    から、個々の形体(shape_aspect)が指す実際の面/エッジのSTEP IDへの対応表を作る。"""
    usage: dict[int, list[int]] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw not in ("GEOMETRIC_ITEM_SPECIFIC_USAGE", "ITEM_IDENTIFIED_REPRESENTATION_USAGE"):
                continue
            fields = _split_top_level_args(args)
            if len(fields) < 5:
                continue
            definition = _ref(fields[2])
            if definition is None:
                continue
            if kw == "GEOMETRIC_ITEM_SPECIFIC_USAGE":
                identified = _ref(fields[4])
                if identified is not None:
                    usage.setdefault(definition, []).append(identified)
            else:
                # identified_item は SET_REPRESENTATION_ITEM((#面, #面...)) の形になることがある
                for identified in _refs_in(fields[4]):
                    usage.setdefault(definition, []).append(identified)
    return usage


def _resolve_shape_aspect_faces(
    shape_aspect_ref: Optional[int],
    rel_map: dict[int, list[int]],
    usage_map: dict[int, list[int]],
    _seen: Optional[set[int]] = None,
) -> list[int]:
    """shape_aspect(複合の場合あり)から、実際に対象となる面/エッジのSTEP IDを再帰的に集める。"""
    if shape_aspect_ref is None:
        return []
    if _seen is None:
        _seen = set()
    if shape_aspect_ref in _seen or len(_seen) > 200:
        return []
    _seen.add(shape_aspect_ref)

    faces = list(usage_map.get(shape_aspect_ref, []))
    for child in rel_map.get(shape_aspect_ref, []):
        faces.extend(_resolve_shape_aspect_faces(child, rel_map, usage_map, _seen))
    return faces


def extract_geometric_tolerances(entities: dict[int, Entity], unit_factor: float = 1.0) -> list[dict]:
    rel_map = _build_shape_aspect_relationships(entities)
    usage_map = _build_geometric_item_usage_map(entities)

    # 公差域(TOLERANCE_ZONE): 対象の幾何公差ID -> 形状名('cylindrical or circular' / 'spherical' ...)
    zone_form_by_tol: dict[int, str] = {}
    for zid, zent in entities.items():
        for kw, args in zent.parts:
            if kw != "TOLERANCE_ZONE":
                continue
            zf = _split_top_level_args(args)
            if len(zf) < 6:
                continue
            form_ref = _ref(zf[5])
            form_name = ""
            if form_ref is not None and form_ref in entities:
                for k2, a2 in entities[form_ref].parts:
                    if k2 == "TOLERANCE_ZONE_FORM":
                        ff = _split_top_level_args(a2)
                        form_name = _unquote(ff[0]) if ff else ""
            for tref in _refs_in(zf[4]):
                zone_form_by_tol[tref] = form_name

    results = []
    for eid, ent in entities.items():
        base = None
        datum_ref_field = None
        modifier_text = ""
        type_kw = None
        for kw, args in ent.parts:
            if kw == "GEOMETRIC_TOLERANCE_WITH_MODIFIERS":
                modifier_text += args
            if kw == "GEOMETRIC_TOLERANCE":
                base = _split_top_level_args(args)
            elif kw == "GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE":
                datum_ref_field = args
            elif kw in TOLERANCE_TYPE_JA:
                type_kw = kw
        if base is None or type_kw is None:
            # 単体エンティティ形式: PARALLELISM_TOLERANCE(name, desc, magnitude, aspect, (datum_system))
            # (複合エンティティではなく、公差型そのものが全属性を持つ書き方)
            for kw, args in ent.parts:
                if kw in TOLERANCE_TYPE_JA and len(ent.parts) == 1:
                    ff = _split_top_level_args(args)
                    if len(ff) >= 4 and _ref(ff[2]) is not None:
                        type_kw = kw
                        base = ff
                        if len(ff) > 4:
                            datum_ref_field = ff[4]
        if base is None or type_kw is None:
            continue
        name = _unquote(base[0]) if len(base) > 0 else ""
        desc = _unquote(base[1]) if len(base) > 1 else ""
        magnitude_ref = _ref(base[2]) if len(base) > 2 else None
        shape_aspect_ref = _ref(base[3]) if len(base) > 3 else None
        value = resolve_measure(entities, magnitude_ref)
        if value is not None:
            value = round(value * unit_factor, 6)

        datums: list[str] = []
        if datum_ref_field is not None:
            for f in _split_top_level_args(datum_ref_field):
                for sub in _refs_in(f):
                    datums.extend(resolve_datum_letters(entities, sub))

        face_refs = sorted(set(_resolve_shape_aspect_faces(shape_aspect_ref, rel_map, usage_map)))

        results.append({
            "id": eid,
            "keyword": type_kw,
            "type_ja": TOLERANCE_TYPE_JA.get(type_kw, type_kw),
            "name": name,
            "description": desc,
            "value_mm": value,
            "datums": datums,
            "modifiers": _modifier_symbols(modifier_text),
            "zone_form": zone_form_by_tol.get(eid, ""),
            "shape_aspect_ref": shape_aspect_ref,
            "face_refs": face_refs,
        })
    results.sort(key=lambda r: r["id"])
    return results


# ---------------------------------------------------------------------------
# 面(ADVANCED_FACE)のアンカー座標抽出
#
# OCCT(OCP)のSTEPエンティティ<->TopoDS_Shape対応表(XSControl_TransferReader)は
# このプロジェクトの環境では機能しなかった(Model().Number()が常に0を返す、
# ShapeResult()は多くの場合Null形状を返す)ため、代わりにSTEPのトポロジ
# (FACE_BOUND -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE -> VERTEX_POINT
# -> CARTESIAN_POINT)を辿って各面の境界頂点の実座標を集め、
# テッセレーション後のメッシュ側の頂点座標と直接突き合わせて面を特定する。
# ---------------------------------------------------------------------------

def _get_cartesian_point(entities: dict[int, Entity], ref_id: Optional[int]) -> Optional[tuple[float, float, float]]:
    if ref_id is None or ref_id not in entities:
        return None
    for kw, args in entities[ref_id].parts:
        if kw == "CARTESIAN_POINT":
            fields = _split_top_level_args(args)
            if len(fields) >= 2:
                v = _parse_floats(fields[1])
                if len(v) >= 3:
                    return (v[0], v[1], v[2])
    return None


def _get_vertex_point(entities: dict[int, Entity], ref_id: Optional[int]) -> Optional[tuple[float, float, float]]:
    if ref_id is None or ref_id not in entities:
        return None
    for kw, args in entities[ref_id].parts:
        if kw == "VERTEX_POINT":
            fields = _split_top_level_args(args)
            if len(fields) >= 2:
                return _get_cartesian_point(entities, _ref(fields[1]))
    return None


def _get_edge_curve_vertices(entities: dict[int, Entity], ref_id: Optional[int]) -> list[tuple[float, float, float]]:
    if ref_id is None or ref_id not in entities:
        return []
    for kw, args in entities[ref_id].parts:
        if kw == "EDGE_CURVE":
            fields = _split_top_level_args(args)
            if len(fields) >= 3:
                pts = []
                for f in (fields[1], fields[2]):
                    p = _get_vertex_point(entities, _ref(f))
                    if p is not None:
                        pts.append(p)
                return pts
    return []


def _get_face_anchor_points(entities: dict[int, Entity], face_ref: int) -> list[tuple[float, float, float]]:
    """ADVANCED_FACE の境界(FACE_BOUND群)から、実座標の頂点集合を集める。"""
    if face_ref not in entities:
        return []
    points: list[tuple[float, float, float]] = []
    for kw, args in entities[face_ref].parts:
        if kw != "ADVANCED_FACE" and kw != "FACE_SURFACE":
            continue
        fields = _split_top_level_args(args)
        if len(fields) < 2:
            continue
        bound_refs = _refs_in(fields[1])
        for bound_ref in bound_refs:
            if bound_ref not in entities:
                continue
            for bkw, bargs in entities[bound_ref].parts:
                if bkw not in ("FACE_BOUND", "FACE_OUTER_BOUND"):
                    continue
                bfields = _split_top_level_args(bargs)
                if len(bfields) < 2:
                    continue
                wire_ref = _ref(bfields[1])
                if wire_ref is None or wire_ref not in entities:
                    continue
                for wkw, wargs in entities[wire_ref].parts:
                    if wkw != "EDGE_LOOP":
                        continue
                    wfields = _split_top_level_args(wargs)
                    if len(wfields) < 2:
                        continue
                    for oe_ref in _refs_in(wfields[1]):
                        if oe_ref not in entities:
                            continue
                        for okw, oargs in entities[oe_ref].parts:
                            if okw != "ORIENTED_EDGE":
                                continue
                            ofields = _split_top_level_args(oargs)
                            if len(ofields) >= 4:
                                edge_ref = _ref(ofields[3])
                                points.extend(_get_edge_curve_vertices(entities, edge_ref))
    # 重複除去(丸め誤差対策で少数第9位で丸め)
    seen = set()
    uniq = []
    for p in points:
        key = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _cartesian_point_of(entities: dict[int, Entity], ref: int):
    """ref が CARTESIAN_POINT ならその座標 (x, y, z) を返す。違えば None。"""
    ent = entities.get(ref)
    if ent is None:
        return None
    for kw, args in ent.parts:
        if kw != "CARTESIAN_POINT":
            continue
        fields = _split_top_level_args(args)
        if len(fields) < 2:
            return None
        try:
            vals = _parse_floats(fields[1])
        except ValueError:
            return None
        if len(vals) >= 3:
            return (vals[0], vals[1], vals[2])
    return None


def extract_face_anchor_points(entities: dict[int, Entity], face_refs: set[int]) -> dict[int, list[tuple[float, float, float]]]:
    """必要な面IDの集合について、境界頂点座標を一括抽出する。"""
    return {fid: _get_face_anchor_points(entities, fid) for fid in face_refs}


def extract_datums(entities: dict[int, Entity]) -> list[dict]:
    results = []
    for eid, ent in entities.items():
        for kw, args in ent.parts:
            if kw == "DATUM" and len(ent.parts) == 1:
                # DATUM_FEATURE等と複合していない、単体の DATUM(...) のみを一覧対象にする
                fields = _split_top_level_args(args)
                if len(fields) >= 5:
                    results.append({
                        "id": eid,
                        "name": _unquote(fields[0]),
                        "letter": _unquote(fields[-1]),
                    })
    results.sort(key=lambda r: r["letter"])
    return results


# ---------------------------------------------------------------------------
# テッセレーションPMI（注記のグラフィック表現＝折れ線の集合）の抽出
# ---------------------------------------------------------------------------

def _parse_floats(field_text: str) -> tuple[float, ...]:
    """'(1.0,2.0,3.0)' のような文字列を (1.0, 2.0, 3.0) に変換する。"""
    inner = field_text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    return tuple(float(x) for x in _split_top_level_args(inner) if x)


def _parse_point_list(list_field: str) -> list[tuple[float, float, float]]:
    """COORDINATES_LISTの座標フィールド '((x,y,z),(x,y,z),...)' をパースする。"""
    inner = list_field.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    points = []
    for group in _split_top_level_args(inner):
        vals = _parse_floats(group)
        if len(vals) >= 3:
            points.append((vals[0], vals[1], vals[2]))
    return points


def _parse_index_tuples(list_field: str) -> list[list[int]]:
    """曲線集合の接続情報 '((1,2),(3,4,5),...)' をパースする(1始まりのインデックス)。"""
    inner = list_field.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    tuples = []
    for group in _split_top_level_args(inner):
        g = group.strip()
        if g.startswith("(") and g.endswith(")"):
            g = g[1:-1]
        idxs = [int(x) for x in _split_top_level_args(g) if x]
        if idxs:
            tuples.append(idxs)
    return tuples


def _get_coordinates_list(entities: dict[int, Entity], ref_id: Optional[int]) -> Optional[list[tuple[float, float, float]]]:
    if ref_id is None or ref_id not in entities:
        return None
    for kw, args in entities[ref_id].parts:
        if kw == "COORDINATES_LIST":
            fields = _split_top_level_args(args)
            if len(fields) >= 3:
                return _parse_point_list(fields[2])
    return None


def _get_placement_transform(entities: dict[int, Entity], ref_id: Optional[int]):
    """AXIS2_PLACEMENT_3D を辿り、局所座標->グローバル座標へ変換する関数を作る。
    見つからない場合は恒等変換(そのまま)を返す。"""
    identity = lambda p: p  # noqa: E731

    if ref_id is None or ref_id not in entities:
        return identity
    placement = None
    for kw, args in entities[ref_id].parts:
        if kw == "AXIS2_PLACEMENT_3D":
            placement = _split_top_level_args(args)
    if placement is None or len(placement) < 4:
        return identity

    origin_ref = _ref(placement[1])
    zdir_ref = _ref(placement[2])
    xdir_ref = _ref(placement[3])

    def get_point(rid):
        if rid is None or rid not in entities:
            return None
        for kw, args in entities[rid].parts:
            if kw == "CARTESIAN_POINT":
                fields = _split_top_level_args(args)
                if len(fields) >= 2:
                    v = _parse_floats(fields[1])
                    if len(v) >= 3:
                        return v
        return None

    def get_dir(rid):
        if rid is None or rid not in entities:
            return None
        for kw, args in entities[rid].parts:
            if kw == "DIRECTION":
                fields = _split_top_level_args(args)
                if len(fields) >= 2:
                    v = _parse_floats(fields[1])
                    if len(v) >= 3:
                        return v
        return None

    origin = get_point(origin_ref) or (0.0, 0.0, 0.0)
    zdir = get_dir(zdir_ref) or (0.0, 0.0, 1.0)
    xdir = get_dir(xdir_ref) or (1.0, 0.0, 0.0)

    def normalize(v):
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-12 else (0.0, 0.0, 1.0)

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def scale(a, s):
        return (a[0] * s, a[1] * s, a[2] * s)

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    z = normalize(zdir)
    x = normalize(sub(xdir, scale(z, dot(xdir, z))))
    y = cross(z, x)

    def transform(p):
        return (
            origin[0] + p[0] * x[0] + p[1] * y[0] + p[2] * z[0],
            origin[1] + p[0] * x[1] + p[1] * y[1] + p[2] * z[1],
            origin[2] + p[0] * x[2] + p[1] * y[2] + p[2] * z[2],
        )

    return transform


def _decode_triangle_strip(strip: list[int]) -> list[tuple[int, int, int]]:
    """OpenGLのTRIANGLE_STRIPと同じ連結三角形帯。隣接三角形ごとに巻き順を反転させる。"""
    triangles = []
    for i in range(len(strip) - 2):
        if i % 2 == 0:
            triangles.append((strip[i], strip[i + 1], strip[i + 2]))
        else:
            triangles.append((strip[i + 1], strip[i], strip[i + 2]))
    return triangles


def _decode_triangle_fan(fan: list[int]) -> list[tuple[int, int, int]]:
    """OpenGLのTRIANGLE_FANと同じ扇形。先頭の頂点を軸に隣接ペアで三角形を作る。"""
    return [(fan[0], fan[i], fan[i + 1]) for i in range(1, len(fan) - 1)]


def _get_triangle_strips(entities: dict[int, Entity], ref_id: Optional[int]):
    """COMPLEX_TRIANGULATED_SURFACE_SET / TRIANGULATED_SURFACE_SET (文字などの
    塗りつぶしグリフ形状)を解決し、(局所座標点リスト, 三角形の頂点インデックス3つ組リスト) を返す。

    complex_triangulated_surface_set のフィールド順は
    (name, coordinates, pnmax, normals, pnindex, triangle_strips, triangle_fans)。
    どちらに実データが入るかはエクスポートしたCADシステムによって異なる
    (例: 一部のファイルは triangle_strips が空で triangle_fans にのみ三角形が入る)ため、
    両方のフィールドを読み、それぞれ正しいアルゴリズムで展開して合算する。"""
    if ref_id is None or ref_id not in entities:
        return None
    for kw, args in entities[ref_id].parts:
        if kw not in ("COMPLEX_TRIANGULATED_SURFACE_SET", "TRIANGULATED_SURFACE_SET"):
            continue
        fields = _split_top_level_args(args)
        if kw == "COMPLEX_TRIANGULATED_SURFACE_SET":
            strips_idx, fans_idx = 5, 6
        else:
            strips_idx, fans_idx = 3, None
        if len(fields) <= strips_idx:
            continue
        coord_ref = _ref(fields[1])
        points = _get_coordinates_list(entities, coord_ref)
        if not points:
            continue
        triangles: list[tuple[int, int, int]] = []
        for strip in _parse_index_tuples(fields[strips_idx]):
            if len(strip) >= 3:
                triangles.extend(_decode_triangle_strip(strip))
        if fans_idx is not None and len(fields) > fans_idx:
            for fan in _parse_index_tuples(fields[fans_idx]):
                if len(fan) >= 3:
                    triangles.extend(_decode_triangle_fan(fan))
        if triangles:
            return points, triangles
    return None


def extract_tessellated_pmi(entities: dict[int, Entity]) -> list[dict]:
    """TESSELLATED_ANNOTATION_OCCURRENCE を辿り、注記のグラフィック表現を
    グローバル3D座標に変換して抽出する。
    - polylines: 引出線・寸法線・公差枠などの輪郭線 (TESSELLATED_CURVE_SET)
    - glyph:     数字・文字などの塗りつぶし形状 (COMPLEX_TRIANGULATED_SURFACE_SET)
    """
    results = []

    for eid, ent in entities.items():
        occ_fields = None
        for kw, args in ent.parts:
            if kw == "TESSELLATED_ANNOTATION_OCCURRENCE":
                occ_fields = _split_top_level_args(args)
        if occ_fields is None or len(occ_fields) < 3:
            continue

        name = _unquote(occ_fields[0])
        item_ref = _ref(occ_fields[2])
        if item_ref is None or item_ref not in entities:
            continue

        item_ent = entities[item_ref]
        placement_ref = None
        curve_set_refs: list[int] = []
        surface_set_refs: list[int] = []
        for kw, args in item_ent.parts:
            if kw == "REPOSITIONED_TESSELLATED_ITEM":
                placement_ref = _ref(args)
            elif kw == "TESSELLATED_GEOMETRIC_SET":
                for f in _split_top_level_args(args):
                    for r in _refs_in(f):
                        target = entities.get(r)
                        if target is None:
                            continue
                        kinds = {k for k, _ in target.parts}
                        if "TESSELLATED_CURVE_SET" in kinds:
                            curve_set_refs.append(r)
                        elif kinds & {"COMPLEX_TRIANGULATED_SURFACE_SET", "TRIANGULATED_SURFACE_SET"}:
                            surface_set_refs.append(r)
            elif kw == "TESSELLATED_CURVE_SET":
                # 複合エンティティに包まれず、item自体が直接カーブセットであるケース
                curve_set_refs.append(item_ref)
            elif kw in ("COMPLEX_TRIANGULATED_SURFACE_SET", "TRIANGULATED_SURFACE_SET"):
                surface_set_refs.append(item_ref)

        transform = _get_placement_transform(entities, placement_ref)

        polylines: list[list[tuple[float, float, float]]] = []
        for cs_ref in curve_set_refs:
            if cs_ref is None or cs_ref not in entities:
                continue
            for kw, args in entities[cs_ref].parts:
                if kw != "TESSELLATED_CURVE_SET":
                    continue
                fields = _split_top_level_args(args)
                if len(fields) < 3:
                    continue
                coord_ref = _ref(fields[1])
                points = _get_coordinates_list(entities, coord_ref)
                if not points:
                    continue
                for idx_tuple in _parse_index_tuples(fields[2]):
                    line = []
                    for idx in idx_tuple:
                        if 1 <= idx <= len(points):
                            line.append(transform(points[idx - 1]))
                    if len(line) >= 2:
                        polylines.append(line)

        glyph_positions: list[float] = []
        glyph_indices: list[int] = []
        for ss_ref in surface_set_refs:
            resolved = _get_triangle_strips(entities, ss_ref)
            if not resolved:
                continue
            points, triangles = resolved
            base = len(glyph_positions) // 3
            for p in points:
                glyph_positions.extend(transform(p))
            for a, b, c in triangles:
                if 1 <= a <= len(points) and 1 <= b <= len(points) and 1 <= c <= len(points):
                    glyph_indices.extend((base + a - 1, base + b - 1, base + c - 1))

        if polylines or glyph_indices:
            entry = {"name": name, "polylines": polylines}
            if glyph_indices:
                entry["glyph"] = {"positions": glyph_positions, "indices": glyph_indices}
            results.append(entry)

    return results


def _callout_names_by_definition(entities: dict[int, Entity]) -> dict[int, list[str]]:
    """DRAUGHTING_MODEL_ITEM_ASSOCIATION(...)(name, description, definition, used_representation, identified_item[, placeholder])
    から、意味側エンティティ(DIMENSIONAL_SIZE / DATUM_FEATURE / 幾何公差 ...) -> 注記(DRAUGHTING_CALLOUT)名 の対応を作る。"""
    out: dict[int, list[str]] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if not kw.startswith("DRAUGHTING_MODEL_ITEM_ASSOCIATION"):
                continue
            f = _split_top_level_args(args)
            if len(f) < 5:
                continue
            definition = _ref(f[2])
            item = _ref(f[4])
            if definition is None or item is None or item not in entities:
                continue
            for k2, a2 in entities[item].parts:
                if k2 == "DRAUGHTING_CALLOUT":
                    ff = _split_top_level_args(a2)
                    nm = _unquote(ff[0]) if ff else ""
                    if nm:
                        out.setdefault(definition, [])
                        if nm not in out[definition]:
                            out[definition].append(nm)
    return out


def _fmt_num(v: float) -> str:
    return f"{v:g}"


def extract_dimensions(entities: dict[int, Entity], unit_factor: float = 1.0) -> list[dict]:
    """DIMENSIONAL_SIZE / DIMENSIONAL_LOCATION (サイズ寸法・位置寸法) を、幾何公差と同じ形式で返す。
    公称値は DIMENSIONAL_CHARACTERISTIC_REPRESENTATION -> SHAPE_DIMENSION_REPRESENTATION の測定値、
    寸法公差は PLUS_MINUS_TOLERANCE(TOLERANCE_VALUE(下限, 上限)) から得る。"""
    rel_map = _build_shape_aspect_relationships(entities)
    usage_map = _build_geometric_item_usage_map(entities)
    callouts = _callout_names_by_definition(entities)

    # 寸法ID -> SHAPE_DIMENSION_REPRESENTATION の項目群から公称値
    nominal: dict[int, float] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw != "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION":
                continue
            f = _split_top_level_args(args)
            dim_ref = _ref(f[0]) if f else None
            rep_ref = _ref(f[1]) if len(f) > 1 else None
            if dim_ref is None or rep_ref is None or rep_ref not in entities:
                continue
            for k2, a2 in entities[rep_ref].parts:
                if k2 != "SHAPE_DIMENSION_REPRESENTATION":
                    continue
                ff = _split_top_level_args(a2)
                if len(ff) < 2:
                    continue
                for item_ref in _refs_in(ff[1]):
                    item = entities.get(item_ref)
                    if item is None:
                        continue
                    val = None
                    label = ""
                    for k3, a3 in item.parts:
                        if k3 == "MEASURE_WITH_UNIT" or k3.endswith("_MEASURE_WITH_UNIT"):
                            m = re.match(r"\s*[A-Z_]+\(([^()]*)\)", _split_top_level_args(a3)[0] if _split_top_level_args(a3) else "")
                            if m:
                                try:
                                    val = float(m.group(1))
                                except ValueError:
                                    pass
                        elif k3 == "REPRESENTATION_ITEM":
                            fa = _split_top_level_args(a3)
                            label = _unquote(fa[0]) if fa else ""
                    if val is not None and (dim_ref not in nominal or "nominal" in label.lower()):
                        nominal[dim_ref] = val

    # 寸法ID -> (下限, 上限)
    tol_range: dict[int, tuple[float, float]] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw != "PLUS_MINUS_TOLERANCE":
                continue
            f = _split_top_level_args(args)
            if len(f) < 2:
                continue
            range_ref, dim_ref = _ref(f[0]), _ref(f[1])
            if range_ref is None or dim_ref is None or range_ref not in entities:
                continue
            for k2, a2 in entities[range_ref].parts:
                if k2 != "TOLERANCE_VALUE":
                    continue
                ff = _split_top_level_args(a2)
                if len(ff) >= 2:
                    lo = resolve_measure(entities, _ref(ff[0]))
                    hi = resolve_measure(entities, _ref(ff[1]))
                    if lo is not None and hi is not None:
                        tol_range[dim_ref] = (lo, hi)

    results = []
    for eid in sorted(entities):
        for kw, args in entities[eid].parts:
            if kw not in ("DIMENSIONAL_SIZE", "DIMENSIONAL_LOCATION"):
                continue
            f = _split_top_level_args(args)
            if kw == "DIMENSIONAL_SIZE":
                aspect_refs = [_ref(f[0])] if f else []
                kind_name = _unquote(f[1]) if len(f) > 1 else ""
            else:  # DIMENSIONAL_LOCATION(name, description, relating, related)
                aspect_refs = [_ref(f[2]), _ref(f[3])] if len(f) > 3 else []
                kind_name = _unquote(f[0]) if f else ""
            names = callouts.get(eid, [])
            face_refs: list[int] = []
            for ar in aspect_refs:
                face_refs.extend(_resolve_shape_aspect_faces(ar, rel_map, usage_map))
            face_refs = sorted(set(face_refs))

            val = nominal.get(eid)
            val_mm = round(val * unit_factor, 6) if val is not None else None
            kn = kind_name.lower()
            if "diameter" in kn:
                layer, icon = "Diameter", "\u2300"
            elif "radius" in kn:
                layer, icon = "Radius", "R"
            elif kw == "DIMENSIONAL_LOCATION":
                layer, icon = "Distance", ""
            else:
                layer, icon = "Width", ""
            text = _fmt_num(val_mm) if val_mm is not None else "?"
            rng = tol_range.get(eid)
            if rng is not None:
                lo, hi = rng[0] * unit_factor, rng[1] * unit_factor
                if abs(abs(lo) - abs(hi)) < 1e-9:
                    text += f" \u00b1{_fmt_num(abs(hi))}"
                else:
                    text += f" +{_fmt_num(hi)}/{_fmt_num(lo)}" if hi >= 0 else f" {_fmt_num(hi)}/{_fmt_num(lo)}"
            results.append({
                "id": eid,
                "keyword": kw,
                "type_ja": "\u5bf8\u6cd5" if not icon else ("\u76f4\u5f84" if layer == "Diameter" else "\u534a\u5f84"),
                "name": names[0] if names else (kind_name or kw),
                "description": kind_name,
                "value_mm": val_mm,
                "datums": [],
                "datum_refs": [],
                "modifiers": "",
                "zone_form": "",
                "shape_aspect_ref": aspect_refs[0] if aspect_refs else None,
                "face_refs": face_refs,
                # 表示用(pmi_display はこれらが既に在ればそのまま使う)
                "layer": layer,
                "symbol_kind": "dimension",
                "dimension_icon": icon,
                "value_text": text,
            })
    return results


def extract_datum_features(entities: dict[int, Entity]) -> dict[str, dict]:
    """データム文字 -> {names: [データム形体記号の注記名...], face_refs: [...]}。
    DATUM_FEATURE(形体) と DATUM(文字) は SHAPE_ASPECT_RELATIONSHIP で結ばれている。"""
    rel_map = _build_shape_aspect_relationships(entities)
    usage_map = _build_geometric_item_usage_map(entities)
    callouts = _callout_names_by_definition(entities)

    letter_by_datum: dict[int, str] = {}
    feature_ids: set[int] = set()
    for eid, ent in entities.items():
        for kw, args in ent.parts:
            if kw == "DATUM" and len(ent.parts) == 1:
                f = _split_top_level_args(args)
                if len(f) >= 5:
                    letter_by_datum[eid] = _unquote(f[-1])
            elif kw == "DATUM_FEATURE":
                feature_ids.add(eid)

    out: dict[str, dict] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw != "SHAPE_ASPECT_RELATIONSHIP":
                continue
            f = _split_top_level_args(args)
            if len(f) < 4:
                continue
            a, b = _ref(f[2]), _ref(f[3])
            for feat, dat in ((a, b), (b, a)):
                if feat in feature_ids and dat in letter_by_datum:
                    d = out.setdefault(letter_by_datum[dat], {"names": [], "face_refs": [], "feature_ids": []})
                    if feat not in d["feature_ids"]:
                        d["feature_ids"].append(feat)
                    for nm in callouts.get(feat, []):
                        if nm not in d["names"]:
                            d["names"].append(nm)
                    d["face_refs"].extend(usage_map.get(feat, []))
    for d in out.values():
        d["face_refs"] = sorted(set(d["face_refs"]))
    return out



def _target_geometry(entities: dict[int, Entity], item_ref: int) -> Optional[dict]:
    """データムターゲット等の幾何要素(点/トリム曲線)を、描画用の点または折れ線にする。"""
    import math
    if item_ref not in entities:
        return None
    for kw, args in entities[item_ref].parts:
        if kw == "CARTESIAN_POINT":
            p = _get_cartesian_point(entities, item_ref)
            return {"point": list(p)} if p else None
        if kw != "TRIMMED_CURVE":
            continue
        f = _split_top_level_args(args)
        if len(f) < 5:
            return None
        pts = []
        for t in (f[2], f[3]):
            for r in _refs_in(t):
                p = _get_cartesian_point(entities, r)
                if p:
                    pts.append(p)
                    break
        if len(pts) != 2:
            return None
        basis = _ref(f[1])
        sense = ".T." in f[4]
        if basis in entities:
            for k2, a2 in entities[basis].parts:
                if k2 == "LINE":
                    return {"polyline": [list(pts[0]), list(pts[1])]}
                if k2 == "CIRCLE":
                    ff = _split_top_level_args(a2)
                    radius = _parse_floats(ff[2])[0] if len(ff) > 2 and _parse_floats(ff[2]) else None
                    tf = _get_placement_transform(entities, _ref(ff[1]))
                    if radius is None:
                        return {"polyline": [list(pts[0]), list(pts[1])]}
                    # 局所座標へ戻す代わりに、中心・軸を直接取得する
                    pl = None
                    for k3, a3 in entities.get(_ref(ff[1]), Entity(0)).parts:
                        if k3 == "AXIS2_PLACEMENT_3D":
                            pl = _split_top_level_args(a3)
                    if pl is None or len(pl) < 3:
                        return {"polyline": [list(pts[0]), list(pts[1])]}
                    c = _get_cartesian_point(entities, _ref(pl[1])) or (0.0, 0.0, 0.0)
                    def dvec(rid, dflt):
                        for k4, a4 in entities.get(rid, Entity(0)).parts:
                            if k4 == "DIRECTION":
                                v = _parse_floats(_split_top_level_args(a4)[1])
                                return v if len(v) >= 3 else dflt
                        return dflt
                    z = dvec(_ref(pl[2]), (0.0, 0.0, 1.0))
                    x = dvec(_ref(pl[3]), (1.0, 0.0, 0.0)) if len(pl) > 3 else (1.0, 0.0, 0.0)
                    zn = math.sqrt(sum(v * v for v in z)) or 1.0
                    z = tuple(v / zn for v in z)
                    d = sum(a * b for a, b in zip(x, z))
                    x = tuple(a - d * b for a, b in zip(x, z))
                    xn = math.sqrt(sum(v * v for v in x)) or 1.0
                    x = tuple(v / xn for v in x)
                    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
                    def ang(p):
                        v = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
                        return math.atan2(sum(a * b for a, b in zip(v, y)), sum(a * b for a, b in zip(v, x)))
                    a1, a2 = ang(pts[0]), ang(pts[1])
                    if not sense:
                        a1, a2 = a2, a1
                    while a2 < a1:
                        a2 += 2 * math.pi
                    n = 24
                    poly = []
                    for i in range(n + 1):
                        a = a1 + (a2 - a1) * i / n
                        poly.append([c[j] + radius * (math.cos(a) * x[j] + math.sin(a) * y[j]) for j in range(3)])
                    if not sense:
                        poly.reverse()
                    return {"polyline": poly}
        return {"polyline": [list(pts[0]), list(pts[1])]}
    return None


NOTE_TOKENS = {
    "#{diameter}": "⌀", "#{plusminus}": "±", "#{degree}": "°", "#{square}": "□",
    "#{profile-of-surface}": "⌓", "#{profile-of-line}": "⌒", "#{position}": "⌖",
    "#{flatness}": "⏥", "#{straightness}": "⏤", "#{parallelism}": "∥",
    "#{perpendicularity}": "⟂", "#{angularity}": "∠", "#{circularity}": "○",
    "#{cylindricity}": "⌭", "#{concentricity}": "◎", "#{symmetry}": "⌯",
    "#{circular-runout}": "↗", "#{total-runout}": "⌰",
}


def _text_by_property_definition(entities: dict[int, Entity]) -> dict[int, str]:
    """PROPERTY_DEFINITION(id) -> ノート等の本文。
    PROPERTY_DEFINITION_REPRESENTATION(definition, representation) -> REPRESENTATION の
    DESCRIPTIVE_REPRESENTATION_ITEM(name, text) を連結する。"""
    out: dict[int, str] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw != "PROPERTY_DEFINITION_REPRESENTATION":
                continue
            f = _split_top_level_args(args)
            if len(f) < 2:
                continue
            definition, rep = _ref(f[0]), _ref(f[1])
            if definition is None or rep is None or rep not in entities:
                continue
            texts: list[str] = []
            for k2, a2 in entities[rep].parts:
                if k2 not in ("REPRESENTATION", "SHAPE_REPRESENTATION"):
                    continue
                ff = _split_top_level_args(a2)
                if len(ff) < 2:
                    continue
                for item in _refs_in(ff[1]):
                    for k3, a3 in entities.get(item, Entity(0)).parts:
                        if k3 == "DESCRIPTIVE_REPRESENTATION_ITEM":
                            fa = _split_top_level_args(a3)
                            if len(fa) >= 2:
                                t = decode_step_string(_unquote(fa[1]))
                                for tok, rep in NOTE_TOKENS.items():
                                    t = t.replace(tok, rep)
                                if t:
                                    texts.append(t)
            if texts:
                out[definition] = "\n".join(texts)
    return out


def extract_annotation_targets(entities: dict[int, Entity], known_names: set[str]) -> list[dict]:
    """特性(公差/寸法/データム)以外の注記(ノート、データムターゲット、ラベル等)について、
    注記名 -> 対応する面(STEP ID)を求める。"""
    rel_map = _build_shape_aspect_relationships(entities)
    usage_map = _build_geometric_item_usage_map(entities)
    callouts = _callout_names_by_definition(entities)

    # データムターゲット(PLACED_DATUM_TARGET_FEATURE)は FEATURE_FOR_DATUM_TARGET_RELATIONSHIP で形体と結ばれる
    target_to_feature: dict[int, list[int]] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw == "FEATURE_FOR_DATUM_TARGET_RELATIONSHIP":
                f = _split_top_level_args(args)
                if len(f) >= 4:
                    a, b = _ref(f[2]), _ref(f[3])
                    if a is not None and b is not None:
                        target_to_feature.setdefault(b, []).append(a)
                        target_to_feature.setdefault(a, []).append(b)

    # データム文字(DATUM)と、データムターゲット(PLACED_DATUM_TARGET_FEATURE / DATUM_TARGET)の関係
    letter_by_datum: dict[int, str] = {}
    target_ids: set[int] = set()
    for eid, ent in entities.items():
        for kw, args in ent.parts:
            if kw == "DATUM" and len(ent.parts) == 1:
                f = _split_top_level_args(args)
                if len(f) >= 5:
                    letter_by_datum[eid] = _unquote(f[-1])
            elif kw in ("PLACED_DATUM_TARGET_FEATURE", "DATUM_TARGET"):
                target_ids.add(eid)
    letter_of_target: dict[int, str] = {}
    for ent in entities.values():
        for kw, args in ent.parts:
            if kw != "SHAPE_ASPECT_RELATIONSHIP":
                continue
            f = _split_top_level_args(args)
            if len(f) < 4:
                continue
            a, b = _ref(f[2]), _ref(f[3])
            for tgt, dat in ((a, b), (b, a)):
                if tgt in target_ids and dat in letter_by_datum:
                    letter_of_target[tgt] = letter_by_datum[dat]

    texts_by_def = _text_by_property_definition(entities)
    text_by_name: dict[str, str] = {}
    by_name: dict[str, list[int]] = {}
    letter_by_name: dict[str, str] = {}
    geom_by_name: dict[str, list[dict]] = {}
    for definition, names in callouts.items():
        for nm in names:
            if nm in known_names:
                continue
            faces = list(_resolve_shape_aspect_faces(definition, rel_map, usage_map))
            for feat in target_to_feature.get(definition, []):
                faces.extend(_resolve_shape_aspect_faces(feat, rel_map, usage_map))
            by_name.setdefault(nm, []).extend(faces)
            if definition in texts_by_def and nm not in text_by_name:
                text_by_name[nm] = texts_by_def[definition]
            for gref in faces:
                g = _target_geometry(entities, gref)
                if g is not None:
                    geom_by_name.setdefault(nm, []).append(g)
            if definition in letter_of_target:
                letter_by_name[nm] = letter_of_target[definition]

    out = []
    for nm, faces in by_name.items():
        low = nm.lower()
        is_target = nm in letter_by_name or low.startswith("datum target")
        kind = "datum_target" if is_target else (
            "note" if low.startswith(("text", "note", "label")) else "other")
        item = {"name": nm, "kind": kind, "face_refs": sorted(set(faces))}
        if nm in letter_by_name:
            item["datum_letter"] = letter_by_name[nm]
        if nm in text_by_name:
            item["text"] = text_by_name[nm]
        geoms = geom_by_name.get(nm, [])
        if geoms:
            item["points"] = [g["point"] for g in geoms if "point" in g]
            item["polylines"] = [g["polyline"] for g in geoms if "polyline" in g]
        out.append(item)
    return out


def extract_saved_views(entities: dict[int, Entity]) -> list[dict]:
    """CAMERA_MODEL_D3 (保存ビュー/カメラ) を取り出す。
    AP242/AP214 では、カメラは配置(AXIS2_PLACEMENT_3D)の -Z 方向を向き、
    +Y(= Z x X)が画面の上方向。"""
    def get_vec(rid, kw_name):
        if rid is None or rid not in entities:
            return None
        for kw, args in entities[rid].parts:
            if kw == kw_name:
                fields = _split_top_level_args(args)
                if len(fields) >= 2:
                    v = _parse_floats(fields[1])
                    if len(v) >= 3:
                        return (v[0], v[1], v[2])
        return None

    def norm(v):
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-12 else None

    views: list[dict] = []
    for eid in sorted(entities):
        for kw, args in entities[eid].parts:
            if kw != "CAMERA_MODEL_D3":
                continue
            f = _split_top_level_args(args)
            if len(f) < 2:
                continue
            placement_ref = _ref(f[1])
            if placement_ref is None or placement_ref not in entities:
                continue
            pl = None
            for k2, a2 in entities[placement_ref].parts:
                if k2 == "AXIS2_PLACEMENT_3D":
                    pl = _split_top_level_args(a2)
            if pl is None or len(pl) < 3:
                continue
            origin = get_vec(_ref(pl[1]), "CARTESIAN_POINT") or (0.0, 0.0, 0.0)
            z = norm(get_vec(_ref(pl[2]), "DIRECTION") or (0.0, 0.0, 1.0))
            xr = norm(get_vec(_ref(pl[3]), "DIRECTION") or (1.0, 0.0, 0.0)) if len(pl) > 3 else (1.0, 0.0, 0.0)
            if z is None or xr is None:
                continue
            # x を z に直交化し、y = z x x
            d = xr[0] * z[0] + xr[1] * z[1] + xr[2] * z[2]
            x = norm((xr[0] - d * z[0], xr[1] - d * z[1], xr[2] - d * z[2]))
            if x is None:
                continue
            y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
            name = _unquote(f[0]) if f else ""
            views.append({
                "name": name or f"Camera #{eid}",
                "source": "step",
                "position": list(origin),
                "direction": [-z[0], -z[1], -z[2]],   # 視線方向(カメラ -> 対象)
                "up": list(y),
            })
    return views


def extract_pmi(step_text: str) -> dict:
    entities = parse_entities(step_text)
    unit_factor = find_length_unit_to_mm_factor(entities)
    tolerances = extract_geometric_tolerances(entities, unit_factor)
    tolerances.extend(extract_dimensions(entities, unit_factor))     # サイズ寸法・位置寸法
    datum_features = extract_datum_features(entities)
    # データム形体の上に設定された特性は、特性欄でデータム三角記号(▶[A])を併記する
    for t in tolerances:
        tf = set(t.get("face_refs", []))
        sa = t.get("shape_aspect_ref")
        t["attached_datums"] = sorted(
            letter for letter, df in datum_features.items()
            if (tf and tf & set(df["face_refs"])) or (sa is not None and sa in df["feature_ids"])
        )
    # 同じデータム形体に サイズ公差(寸法) と 幾何公差 の両方があるときは、幾何公差の方だけに三角記号を書く
    letters_with_geo = {
        letter for t in tolerances if t.get("symbol_kind") != "dimension" for letter in t["attached_datums"]
    }
    for t in tolerances:
        if t.get("symbol_kind") == "dimension":
            t["attached_datums"] = [l for l in t["attached_datums"] if l not in letters_with_geo]
    known_names = {t["name"] for t in tolerances}
    for _df in datum_features.values():
        known_names.update(_df["names"])
    annotations = extract_annotation_targets(entities, known_names)

    # 面の境界頂点座標(ハイライト用マッチングのアンカー)は、B-repのローカル座標系と
    # 同じグローバル単位系(unit_factor)に従うため、ここでもmmへ換算する。
    # (テッセレーションPMIの描画座標は対象外。find_length_unit_to_mm_factorの注記を参照)
    all_face_refs: set[int] = set()
    for tol in tolerances:
        all_face_refs.update(tol.get("face_refs", []))
    for df in datum_features.values():
        all_face_refs.update(df["face_refs"])
    for an in annotations:
        all_face_refs.update(an["face_refs"])
    anchor_points = extract_face_anchor_points(entities, all_face_refs)
    for tol in tolerances:
        pts: list[tuple[float, float, float]] = []
        for fid in tol.get("face_refs", []):
            pts.extend(anchor_points.get(fid, []))
        tol["anchor_points"] = [[p[0] * unit_factor, p[1] * unit_factor, p[2] * unit_factor] for p in pts]
        # 寸法が面ではなく CARTESIAN_POINT(点間距離)を参照している場合は、その座標を返す
        cps = []
        for fid in tol.get("face_refs", []):
            cp = _cartesian_point_of(entities, fid)
            if cp is not None:
                cps.append([cp[0] * unit_factor, cp[1] * unit_factor, cp[2] * unit_factor])
        if cps:
            tol["points"] = cps

    datums = extract_datums(entities)
    for d in datums:
        df = datum_features.get(d["letter"])
        d["names"] = df["names"] if df else []
        pts = []
        for fid in (df["face_refs"] if df else []):
            pts.extend(anchor_points.get(fid, []))
        d["anchor_points"] = [[p[0] * unit_factor, p[1] * unit_factor, p[2] * unit_factor] for p in pts]

    for an in annotations:
        pts = []
        for fid in an["face_refs"]:
            pts.extend(anchor_points.get(fid, []))
        an["anchor_points"] = [[p[0] * unit_factor, p[1] * unit_factor, p[2] * unit_factor] for p in pts]
        del an["face_refs"]

    return {
        "geometric_tolerances": tolerances,
        "annotations": annotations,
        "datums": datums,
        "tessellated": extract_tessellated_pmi(entities),
        "saved_views": extract_saved_views(entities),
        "unit_factor_to_mm": unit_factor,
    }


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    result = extract_pmi(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
