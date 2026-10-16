"""幾何公差リスト(左パネル)用の表示情報。3DA Balloon & Mirror Viewer の
ISO 1101 表記(designator + feature control frame)に合わせた項目を付与する。"""
from __future__ import annotations

import re

# STEPキーワード -> (英名の種別 = qif_viewer の type_name, symbol_kind)
STEP_KIND = {
    "FLATNESS_TOLERANCE": ("Flatness", "flatness"),
    "STRAIGHTNESS_TOLERANCE": ("Straightness", "straightness"),
    "ROUNDNESS_TOLERANCE": ("Circularity", "circularity"),
    "CIRCULARITY_TOLERANCE": ("Circularity", "circularity"),
    "CYLINDRICITY_TOLERANCE": ("Cylindricity", "cylindricity"),
    "PARALLELISM_TOLERANCE": ("Parallelism", "parallelism"),
    "PERPENDICULARITY_TOLERANCE": ("Perpendicularity", "perpendicularity"),
    "ANGULARITY_TOLERANCE": ("Angularity", "angularity"),
    "POSITION_TOLERANCE": ("Position", "position"),
    "CONCENTRICITY_TOLERANCE": ("Concentricity", "concentricity"),
    "COAXIALITY_TOLERANCE": ("Concentricity", "concentricity"),
    "SYMMETRY_TOLERANCE": ("Symmetry", "symmetry"),
    "CIRCULAR_RUNOUT_TOLERANCE": ("CircularRunout", "circular_runout"),
    "TOTAL_RUNOUT_TOLERANCE": ("TotalRunout", "total_runout"),
    "SURFACE_PROFILE_TOLERANCE": ("SurfaceProfile", "profile_surface"),
    "PROFILE_OF_SURFACE_TOLERANCE": ("SurfaceProfile", "profile_surface"),
    "LINE_PROFILE_TOLERANCE": ("LineProfile", "profile_line"),
    "PROFILE_OF_LINE_TOLERANCE": ("LineProfile", "profile_line"),
}


def decorate_step_tolerances(tolerances: list[dict]) -> None:
    for i, t in enumerate(tolerances, start=1):
        if t.get("symbol_kind"):          # 寸法など、抽出時に表示項目が決まっているもの
            m = re.search(r"\((\d+)\)", t.get("name") or "")
            t["designator"] = m.group(1) if m else f"{i:02d}"
            continue
        layer, kind = STEP_KIND.get(t.get("keyword", ""), (t.get("type_ja", "Other"), "generic"))
        t["layer"] = layer
        t["symbol_kind"] = kind
        m = re.search(r"\((\d+)\)", t.get("name") or "")
        t["designator"] = m.group(1) if m else f"{i:02d}"
        v = t.get("value_mm")
        text = f"{v:g}" if isinstance(v, (int, float)) else "?"
        form = (t.get("zone_form") or "").lower()
        if isinstance(v, (int, float)):
            if form.startswith("spherical"):
                text = "S⌀" + text      # S⌀
            elif form.startswith("cylindrical") or form.startswith("circular"):
                text = "⌀" + text       # ⌀
            text += t.get("modifiers") or ""
        t["value_text"] = text
        t["datum_refs"] = list(t.get("datums") or [])
        t["dimension_icon"] = ""
