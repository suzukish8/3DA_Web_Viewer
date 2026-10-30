"""Readable text dump of a JT file's decoded PMI (shown in the source panel,
since the JT file itself is binary)."""
from __future__ import annotations

from pathlib import Path

from .pmi import extract_pmi
from .reader import SEGMENT_NAMES, read_jt
from .semantic import to_semantic


def jt_report_text(path: str, doc=None) -> str:
    jt = read_jt(path)
    sem = to_semantic(extract_pmi(jt))
    out = [f"# {Path(path).name}", f"# {jt.version}", ""]
    counts: dict[str, int] = {}
    for s in jt.segments:
        name = SEGMENT_NAMES.get(s.type, f"type {s.type}")
        counts[name] = counts.get(name, 0) + 1
    out.append("Segments: " + ", ".join(f"{n} x{c}" for n, c in sorted(counts.items())))
    if doc is not None:
        out.append("")
        out.extend(doc.warnings)
    out += ["", f"characteristics={len(sem.characteristics)} dimensions={len(sem.dimensions)} "
            f"datums={len(sem.datums)} notes/symbols={len(sem.notes)}", ""]
    for c in sem.characteristics:
        ds = "|".join(",".join(d.label + (f"({d.modifier[0]})" if d.modifier else "") for d in lvl)
                      for lvl in (c.primary, c.secondary, c.tertiary) if lvl)
        out.append(f"FCF  {c.type_name:<17} {c.tolerance_value!s:>6} {c.zone_shape or '':<11}"
                   f" {c.material_condition or '':<8} datums={ds or '-'}")
    for d in sem.dimensions:
        out.append(f"DIM  value={d.value} text={d.text!r} +{d.upper_delta}/{d.lower_delta}")
    for d in sem.datums:
        tag = f"target#{d.target_index}" if d.is_target else "feature"
        out.append(f"DAT  {d.label} ({tag})")
    for n in sem.notes:
        out.append(f"{n.kind[:4].upper():<4} {n.text!r}")
    return "\n".join(out) + "\n"
