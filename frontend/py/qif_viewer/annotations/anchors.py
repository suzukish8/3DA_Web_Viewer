"""Resolves a FeatureNominal to a 3D anchor (point, direction) for annotation
placement. Composite features (kind == "Group") resolve recursively by
averaging their referenced sub-features' anchors.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from qif_viewer.model.brep import CylinderSurface, Face, PlaneSurface
from qif_viewer.model.document import QifDocument
from qif_viewer.model.features import CylinderFeatureNominal, FeatureNominal, GenericFeatureNominal, PlaneFeatureNominal
from qif_viewer.parsing.xml_utils import id_list
from qif_viewer.tessellate.curves import rodrigues_rotate
from qif_viewer.tessellate.geometry_utils import loop_chain_tolerance
from qif_viewer.tessellate.loops import loop_to_polyline
from qif_viewer.tessellate.surfaces import tessellate_face

Anchor = tuple[np.ndarray, np.ndarray]  # (point, direction)


def _outer_loop_vertex_centroid(face, doc: QifDocument) -> Optional[np.ndarray]:
    """Average of a face's outer-loop boundary vertices -- a point
    guaranteed to sit within its actual trimmed region (for any face
    that isn't a wild non-convex shape). Faces are commonly trimmed to a
    small region of their underlying (infinite, untrimmed) surface, so a
    reference point tied only to the *surface*'s own parametrization can
    legitimately land nowhere near where the face actually is."""
    loop_id = next(
        (lid for lid in face.loop_ids if (loop := doc.loops.get(lid)) is not None and loop.form == "OUTER"),
        face.loop_ids[0] if face.loop_ids else None,
    )
    loop = doc.loops.get(loop_id) if loop_id is not None else None
    if loop is None:
        return None
    points = []
    for coedge in loop.coedges:
        edge = doc.edges.get(coedge.edge_id)
        if edge is None:
            continue
        for vid in (edge.vertex_beg_id, edge.vertex_end_id):
            vertex = doc.vertices.get(vid)
            if vertex is not None:
                points.append(vertex.point)
    return np.mean(points, axis=0) if points else None


def _planar_interior_point(face: Face, surface: PlaneSurface, doc: QifDocument) -> Optional[np.ndarray]:
    """A point guaranteed to sit within a planar face's actual trimmed
    material, unlike _outer_loop_vertex_centroid's plain vertex average --
    that average is only guaranteed to fall inside the trimmed region for a
    convex boundary. A non-convex (e.g. L-shaped, or notched) face's outer
    loop can average out to a point in the *missing* material -- the
    "notch" -- landing the leader line's anchor well off the part with no
    warning, since the point still satisfies "on the face's own plane."

    Reuses the exact same triangulation the renderer itself uses
    (tessellate/surfaces/planar.py, via the shared tessellate_face
    dispatcher) and takes the largest triangle's own centroid, which -- as
    a triangle, always convex -- is guaranteed to sit inside the solid."""
    tol = loop_chain_tolerance(doc)
    loop_polylines: list[np.ndarray] = []
    loop_forms: list[str] = []
    local_warnings: list[str] = []
    for loop_id in face.loop_ids:
        loop = doc.loops.get(loop_id)
        if loop is None:
            continue
        poly = loop_to_polyline(loop, doc, local_warnings, tol=tol)
        if len(poly) >= 2:
            loop_polylines.append(poly)
            loop_forms.append(loop.form)
    if not loop_polylines:
        return None
    soup = tessellate_face(face, surface, loop_polylines, loop_forms, local_warnings)
    if soup.triangles is None or len(soup.triangles) == 0:
        return None
    p0 = soup.points[soup.triangles[:, 0]]
    p1 = soup.points[soup.triangles[:, 1]]
    p2 = soup.points[soup.triangles[:, 2]]
    areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    best = int(np.argmax(areas))
    return (p0[best] + p1[best] + p2[best]) / 3.0


def _face_anchor(face_id: str, doc: QifDocument) -> Optional[Anchor]:
    """A point-and-outward-direction anchor taken directly from a B-Rep
    face's own (planar or cylindrical) surface. Used whenever a
    FeatureNominal's own <Location> is a computed reference point that
    doesn't actually sit on any real surface -- e.g. a feature spanning
    several faces (an OppositeParallelPlanes' two opposing walls, or a
    SurfaceProfile over two side flanks) typically has its <Location>
    authored as their average/centroid, which commonly lands in the gap
    *between* the faces rather than on any of them -- sending the leader
    line into thin air instead of touching the part.

    The point itself is always the face's own outer-loop vertex centroid,
    not the underlying surface's own <Origin>/<AxisPoint> -- those are
    just arbitrary reference points for the surface's parametrization,
    and can themselves fall well outside where the face is actually
    trimmed to (confirmed empirically: a plane's own Origin sitting on
    the model's center axis while the face it backs is trimmed to a
    narrow strip well off to one side; likewise a cylinder's AxisPoint
    can sit anywhere along the axis, including well past the trimmed
    end of a short, bounded cylindrical face).

    The one exception is a cylindrical face's *radial* position: the
    outer loop's vertex centroid isn't used for that, because a face
    whose trim spans (close to) a symmetric arc -- e.g. a short ring-
    shaped chamfer face, front and back edges mirrored across the axis --
    has vertices that average out to zero radial offset, landing back
    exactly *on* the axis (the same "averaging cancels out the real
    position" failure as everywhere else here, just reached a different
    way). The face's own domain_angle midpoint gives a reliable in-bounds
    angle instead."""
    face = doc.faces.get(face_id)
    if face is None:
        return None
    surface = doc.surfaces.get(face.surface_id)
    centroid = _outer_loop_vertex_centroid(face, doc)

    if isinstance(surface, PlaneSurface):
        direction = np.cross(surface.dir_u, surface.dir_v)
        point = _planar_interior_point(face, surface, doc)
        if point is None:
            point = centroid if centroid is not None else surface.origin
    elif isinstance(surface, CylinderSurface):
        if centroid is None:
            return None
        axis_dir = surface.axis_dir / np.linalg.norm(surface.axis_dir)
        # Axial position from the centroid (reliable -- it's the radial
        # component that cancels out, not this one), radial direction
        # from the middle of the face's own trimmed angular range.
        axial_offset = np.dot(centroid - surface.axis_point, axis_dir)
        axis_point_here = surface.axis_point + axial_offset * axis_dir
        mid_angle = (surface.domain_angle[0] + surface.domain_angle[1]) / 2
        direction = rodrigues_rotate(surface.dir_beg, axis_dir, mid_angle)
        point = axis_point_here + (surface.diameter / 2) * direction
    else:
        return None  # e.g. a cone or unsupported surface -- caller falls back

    if face.turned:
        direction = -direction
    norm = np.linalg.norm(direction)
    if norm < 1e-9 or point is None:
        return None
    return point, direction / norm


def resolve_anchor(feature_nominal: FeatureNominal, doc: QifDocument, _seen: Optional[set] = None) -> Optional[Anchor]:
    # Always prefer a point taken directly from the referenced B-Rep
    # face's own actual (trimmed) surface over the FeatureNominal's own
    # authored <Location> -- even for a single-face reference. <Location>
    # is a file-authored reference point for the feature as a whole, not
    # guaranteed to sit within that one face's actual trim boundary (e.g.
    # a Flatness's <Location> landing just past the edge of a narrow
    # strip face); _face_anchor derives the point from the real tessellated
    # geometry instead, so it can't miss the material. resolve_face_ids
    # already knows how to recurse into a Group to find every face a
    # multi-face FeatureNominal ultimately touches, so reuse it here rather
    # than re-deriving the same traversal. Tries each face in turn since
    # not every surface type is supported (e.g. a cone) -- the first one
    # that resolves wins; <Location>/axis fields below are the fallback
    # for when no face anchor could be resolved at all.
    face_ids = resolve_face_ids(feature_nominal, doc)
    for face_id in face_ids:
        anchor = _face_anchor(face_id, doc)
        if anchor is not None:
            return anchor

    if isinstance(feature_nominal, PlaneFeatureNominal):
        return feature_nominal.location, feature_nominal.normal

    if isinstance(feature_nominal, CylinderFeatureNominal):
        return feature_nominal.axis_point, feature_nominal.axis_dir

    if isinstance(feature_nominal, GenericFeatureNominal):
        if feature_nominal.location is not None:
            direction = feature_nominal.axis_dir if feature_nominal.axis_dir is not None else np.array([0.0, 0.0, 1.0])
            return feature_nominal.location, direction

        if feature_nominal.kind == "Group":
            seen = _seen if _seen is not None else set()
            if feature_nominal.id in seen:
                return None
            seen.add(feature_nominal.id)
            sub_ids = id_list(feature_nominal.raw_element, "FeatureNominalIds")
            points: list[np.ndarray] = []
            directions: list[np.ndarray] = []
            for sub_id in sub_ids:
                sub = doc.feature_nominals.get(sub_id)
                if sub is None:
                    continue
                anchor = resolve_anchor(sub, doc, seen)
                if anchor is not None:
                    points.append(anchor[0])
                    directions.append(anchor[1])
            if not points:
                return None
            centroid = np.mean(points, axis=0)
            direction = np.mean(directions, axis=0) if directions else np.array([0.0, 0.0, 1.0])
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
            return centroid, direction

        return None  # e.g. OtherSurfaceFeatureNominal with no usable anchor at all

    return None


def dimension_face_options(
    face_ids: list[str], doc: QifDocument,
    pair: tuple[np.ndarray, np.ndarray],
) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
    """The distinct *faces* a size dimension of a hole / slot / rib through
    a plate could be attached to, for the right-click "Set Dimension
    Face..." dialog (ui/main_window.py) -- the user picks; nothing here
    guesses. `face_ids` is the dimension's own resolved B-Rep faces
    (Annotation.face_ids); `pair` is resolve_dimension_endpoints()'s own
    centre-of-feature result; each option is that pair slid rigidly onto
    one axial end (rim) of those faces, labelled by that end's coordinate
    along its dominant world axis (e.g. "X = 0.000").

    Returns [] -- meaning "no face to choose, the dialog item is hidden" --
    unless every referenced face is a cylinder (a bored hole) or a plane
    (opposing walls), and the through-thickness extent is < half the
    measured span (a genuine thin wall, not two walls of a solid block).
    Exactly two options for a normal two-sided feature."""
    face_ids = list(dict.fromkeys(face_ids or []))
    faces = []
    for face_id in face_ids:
        face = doc.faces.get(face_id)
        surface = doc.surfaces.get(face.surface_id) if face else None
        if surface is None:
            return []
        faces.append(surface)
    if not faces:
        return []

    boundary: list[np.ndarray] = []
    for face_id in face_ids:
        boundary.extend(_face_boundary_points(face_id, doc))
    if not boundary:
        return []

    p1, p2 = np.asarray(pair[0], dtype=float), np.asarray(pair[1], dtype=float)
    span_len = float(np.linalg.norm(p2 - p1))

    through_dir: Optional[np.ndarray] = None
    if all(isinstance(s, CylinderSurface) for s in faces):
        axis = None
        for s in faces:
            n = np.linalg.norm(s.axis_dir)
            if n < 1e-9:
                return []
            d = np.asarray(s.axis_dir, dtype=float) / n
            if axis is None:
                axis = d
            elif abs(abs(float(np.dot(axis, d))) - 1.0) > 1e-3:
                return []
        through_dir = axis
    elif all(isinstance(s, PlaneSurface) for s in faces) and span_len > 1e-9:
        span_dir = (p2 - p1) / span_len
        seed = np.array([1.0, 0.0, 0.0]) if abs(span_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = seed - float(np.dot(seed, span_dir)) * span_dir
        u /= np.linalg.norm(u)
        v = np.cross(span_dir, u)
        arr = np.asarray(boundary, dtype=float)
        range_u = float(np.dot(arr, u).max() - np.dot(arr, u).min())
        range_v = float(np.dot(arr, v).max() - np.dot(arr, v).min())
        through_dir, through_range = (u, range_u) if range_u <= range_v else (v, range_v)
        if through_range >= 0.5 * span_len:
            return []
    if through_dir is None:
        return []

    proj = [float(np.dot(np.asarray(b, dtype=float) - p1, through_dir)) for b in boundary]
    lo, hi = min(proj), max(proj)
    if abs(hi - lo) < 1e-6:
        return []
    mid = float(np.dot((p1 + p2) / 2.0 - p1, through_dir))
    axis_letter = "XYZ"[int(np.argmax(np.abs(through_dir)))]
    options = []
    for end in (lo, hi):
        shift = (end - mid) * through_dir
        q1, q2 = p1 + shift, p2 + shift
        coord = float((q1 + q2)[int(np.argmax(np.abs(through_dir)))] / 2.0)
        options.append((f"{axis_letter} = {coord:.3f}", (q1, q2)))
    return options


def resolve_dimension_endpoints(
    feature_nominal_ids: list[str], doc: QifDocument, nominal_value: Optional[float] = None,
    type_name: Optional[str] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The two points a linear "size" dimension (Width/Length/Diameter)
    actually spans between -- e.g. a Width's two opposing planar walls, or
    a Diameter's two diametrically-opposite cylindrical rim points -- for
    rendering a proper two-arrow dimension line with extension lines
    (ISO 129/ASME Y14.5), instead of a single leader and arrow pointing at
    just one of them.

    Resolves every face any of the given FeatureNominals ultimately touches
    (recursing through Groups, via resolve_face_ids) to its own on-surface
    point via _face_anchor, then returns whichever *pair* of those points
    best represents the actual measured span. When the characteristic's own
    nominal (basic) size is known, that's the pair whose measured distance
    is closest to it -- e.g. a slot's Width (measured straight across its
    short axis, ~nominal) references the same faces as its Length (the
    far-apart rounded ends, along the long axis instead) would, so
    "farthest apart" alone can't tell those two spans apart; matching
    against the nominal can. Falls back to the farthest pair when no
    nominal is available (e.g. a reference dimension with no toleranced
    target).

    A pair whose two faces have (nearly) opposite outward normals -- the
    signature of a genuine "opposite parallel planes" Width/Length, e.g. a
    slot's two straight side walls -- is measured by *projecting* onto
    that shared normal, not by the two points' raw straight-line distance:
    each face's own outer-loop-vertex-centroid isn't guaranteed to line up
    exactly across from the other's along the slot's length, so the raw
    distance picks up that lengthwise offset too, overstating the actual
    (perpendicular, nominal-matching) gap between the planes. Projecting
    also fixes the returned second point to sit exactly along the normal
    from the first -- guaranteeing the rendered dimension line is
    perpendicular to both faces, per ISO 129/ASME Y14.5, rather than
    running diagonally between two not-quite-opposite sample points. Any
    other pair (e.g. a Length's two rounded slot ends, whose own normals
    are radial -- perpendicular to the length axis being measured, not
    parallel to it -- so projecting onto either would measure the wrong
    thing entirely) still uses the raw straight-line distance, same as
    before. None if fewer than 2 points resolve.

    ``type_name == "Diameter"`` (see _resolve_dimension_pair) bypasses all
    of the above -- a Diameter never needs a SECOND independently-resolved
    face at all, since its far end is always exactly the mirror of the
    first through the feature's own cylinder axis."""
    resolved = _resolve_dimension_pair(feature_nominal_ids, doc, nominal_value, type_name)
    return resolved[0] if resolved is not None else None


def resolve_dimension_end_normal_sign(
    feature_nominal_ids: list[str], doc: QifDocument, nominal_value: Optional[float] = None,
    type_name: Optional[str] = None,
) -> int:
    """Which way the *outward* surface normal at resolve_dimension_
    endpoints()'s first point (p1) runs relative to the p1->p2 span: -1 if
    it points away from p2 (a solid slab/boss -- p1's face looks outward,
    away from the material between the two), +1 if it points toward p2 (a
    slot/hole -- the material is *outside* the pair, so each wall looks
    into the void), 0 if the two faces aren't an opposite-parallel pair
    (no single sign describes both ends, e.g. a Length between a slot's
    two rounded ends).

    When non-zero, p2's outward normal is exactly the opposite of p1's, so
    the caller (ui/viewport.py's per-frame "balloon" occlusion pass) only
    needs this one sign plus the live span direction to know which way each
    end's own face is looking at any moment -- including after a drag/
    right-click override moves p1/p2 -- without re-touching the QIF. Needed
    because Annotation.anchor_direction is just *one* face's normal, which
    says nothing about the *other* end a leader may actually be drawn from."""
    resolved = _resolve_dimension_pair(feature_nominal_ids, doc, nominal_value, type_name)
    return resolved[1] if resolved is not None else 0


def _resolve_dimension_pair(
    feature_nominal_ids: list[str], doc: QifDocument, nominal_value: Optional[float],
    type_name: Optional[str] = None,
) -> Optional[tuple[tuple[np.ndarray, np.ndarray], int]]:
    """Shared body of resolve_dimension_endpoints/resolve_dimension_end_
    normal_sign -- see the former's docstring for the pairing rules.
    Returns ((p1, p2), end_normal_sign)."""
    points: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        for face_id in resolve_face_ids(fn, doc):
            anchor = _face_anchor(face_id, doc)
            if anchor is not None:
                points.append(anchor[0])
                directions.append(anchor[1])
    if not points:
        return None

    if type_name == "Diameter":
        # ユーザー指示（2026-09-28）: Designator 10 以外の Diameter に
        # 寸法線が出ない -- 未分割の円筒面（穴の壁が1枚の B-Rep パッチ）
        # は resolve_face_ids が1点しか返さず、下の「2点の組」探索に
        # 必要な最低2点に届かず size_endpoints が None のまま単純な
        # リーダー表示に落ちていた（複数パッチに分割された、たまたま
        # 中点方向がほぼ正対する穴だけが寸法線を得ていた）。Diameter は
        # そもそも1つの円筒軸の性質なので、2枚目の面を独立に探す必要が
        # 一切ない -- その軸を挟んで最初の点を鏡映するだけで、常に軸を
        # 正確に通る対辺点が求まる。resolve_cylinder_axis が軸を解決
        # できる限り、Diameter は必ずこちらを使う（下の汎用ペア探索より
        # 優先、かつ1点あれば十分）。
        cyl = resolve_cylinder_axis(feature_nominal_ids, doc)
        if cyl is not None:
            axis_point, axis_dir, _radius = cyl
            p1, d1 = points[0], directions[0]
            axial_offset = float(np.dot(p1 - axis_point, axis_dir))
            axis_point_here = axis_point + axial_offset * axis_dir
            p2 = 2.0 * axis_point_here - p1
            measured = float(np.dot(p2 - p1, d1))
            sign = 1 if measured > 1e-9 else (-1 if measured < -1e-9 else 0)
            return (p1, p1 + d1 * measured), sign

    if len(points) < 2:
        return None
    best: Optional[tuple[tuple[np.ndarray, np.ndarray], int]] = None
    best_score = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            sign = 0
            if np.dot(directions[i], directions[j]) < -0.9:
                measured = float(np.dot(points[j] - points[i], directions[i]))
                candidate = (points[i], points[i] + directions[i] * measured)
                dist = abs(measured)
                if dist > 1e-9:
                    sign = 1 if measured > 0 else -1
            else:
                candidate = (points[i], points[j])
                dist = float(np.linalg.norm(points[j] - points[i]))
            score = abs(dist - nominal_value) if nominal_value is not None else -dist
            if best_score is None or score < best_score:
                best_score = score
                best = (candidate, sign)
    return best


def resolve_cylinder_axis(
    feature_nominal_ids: list[str], doc: QifDocument
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """(axis_point, axis_dir, radius) of the first CylinderSurface backing
    any of the given FeatureNominals -- used for a Radius characteristic,
    whose leader must stay perpendicular to (i.e. radial from) the target
    cylindrical surface at all times, including while its anchor is being
    dragged around the circumference (ui/viewport.py's
    _update_annotation_drag). Unlike resolve_anchor/_face_anchor's single
    point+direction (one fixed spot on the circle, picked once from the
    face's own trimmed angular range), this keeps the whole axis/radius so
    a caller can derive a *different* point around the same circumference
    later and still land exactly on the surface, always perpendicular to
    it there too -- not just at the one point originally picked.

    axis_dir is unit-length; axis_point is an arbitrary point on the
    (infinite) axis line, not adjusted for any particular cross-section --
    same "arbitrary reference point for the surface's own parametrization"
    caveat _face_anchor's own docstring already notes for
    CylinderSurface.axis_point (a caller that needs a specific
    cross-section, e.g. the one through some already-resolved anchor
    point, projects onto this axis itself). None if no referenced feature
    resolves to an actual CylinderSurface (e.g. a Radius on some other
    geometry this app doesn't model that way)."""
    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        for face_id in resolve_face_ids(fn, doc):
            face = doc.faces.get(face_id)
            if face is None:
                continue
            surface = doc.surfaces.get(face.surface_id)
            if isinstance(surface, CylinderSurface):
                norm = np.linalg.norm(surface.axis_dir)
                if norm < 1e-9:
                    continue
                return surface.axis_point, surface.axis_dir / norm, surface.diameter / 2
    return None


def _face_boundary_points(face_id: str, doc: QifDocument) -> list[np.ndarray]:
    """Every boundary vertex of a face's loops (outer and inner) -- a cheap
    stand-in for the face's own extent, enough to measure how far the face
    reaches along a given axis. Unlike _outer_loop_vertex_centroid this
    keeps the individual points rather than averaging them."""
    face = doc.faces.get(face_id)
    if face is None:
        return []
    points: list[np.ndarray] = []
    for loop_id in face.loop_ids:
        loop = doc.loops.get(loop_id)
        if loop is None:
            continue
        for coedge in loop.coedges:
            edge = doc.edges.get(coedge.edge_id)
            if edge is None:
                continue
            for vid in (edge.vertex_beg_id, edge.vertex_end_id):
                vertex = doc.vertices.get(vid)
                if vertex is not None:
                    points.append(vertex.point)
    return points


def _face_loops_2d(face_id: str, doc: QifDocument, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray) -> list[np.ndarray]:
    """Every loop of a face (outer and any inner) as a closed 2D polyline
    in the (u_axis, v_axis) basis relative to `origin` -- i.e. the face's
    actual trimmed boundary, arcs tessellated (via loop_to_polyline, the
    same chaining the renderer uses). Used to find where a ray from the
    anchor, travelling in the face's own plane, first leaves the face."""
    face = doc.faces.get(face_id)
    if face is None:
        return []
    tol = loop_chain_tolerance(doc)
    warnings: list[str] = []
    loops_2d: list[np.ndarray] = []
    for loop_id in face.loop_ids:
        loop = doc.loops.get(loop_id)
        if loop is None:
            continue
        poly = loop_to_polyline(loop, doc, warnings, tol=tol)
        if len(poly) < 3:
            continue
        rel = poly - origin
        loops_2d.append(np.column_stack((rel @ u_axis, rel @ v_axis)))
    return loops_2d


def _ray_polyline_first_t(p: tuple[float, float], d: tuple[float, float], loops_2d: list[np.ndarray]) -> Optional[float]:
    """Smallest t > 0 at which the ray p + t*d (d a unit 2D direction)
    crosses any edge of any closed polyline in `loops_2d`. None if it never
    does. Used to stop a sub-anchor at the face's own edge instead of
    letting it fly off into space."""
    px, py = p
    dx, dy = d
    best: Optional[float] = None
    for poly in loops_2d:
        n = len(poly)
        for k in range(n):
            ax, ay = poly[k]
            bx, by = poly[(k + 1) % n]
            ex, ey = bx - ax, by - ay
            denom = dx * ey - dy * ex
            if abs(denom) < 1e-12:
                continue
            apx, apy = ax - px, ay - py
            t = (apx * ey - apy * ex) / denom
            s = (apx * dy - apy * dx) / denom
            if t > 1e-7 and -1e-9 <= s <= 1.0 + 1e-9:
                if best is None or t < best:
                    best = t
    return best


def resolve_sub_anchor_recipe(
    feature_nominal_ids: list[str], doc: QifDocument
) -> Optional[dict]:
    """The fixed *geometry* a "Balloon & Mirror" annotation's sub-anchor
    points are derived from (see Annotation.sub_anchor_recipe). Call
    sub_anchor_points(recipe, live_anchor) to turn it into 3D points.

    No anchor is generated here: sub-anchors are always derived from the
    existing (live) anchor passed to sub_anchor_points.

    Two shapes:
    * ``{"kind": "cylinder", "axis_point", "axis_dir" (unit), "proj_min",
      "proj_max"}`` -- axial extent of the (first) referenced cylindrical
      face along its own axis. sub_anchor_points makes 2 points, moving
      only the coordinate along axis_dir's dominant world axis to each
      end (every other coordinate stays equal to the live anchor's).
    * ``{"kind": "planes", "axes": [dir_u, dir_v] (unit),
      "faces": [{"origin", "normal" (unit), "loops_2d": [Nx2, ...]},
      ...]}`` -- the two in-plane axes of an opposite-parallel-planes pair
      (perpendicular to the dimension-line axis = the shared plane
      normal), plus each face's own trimmed boundary as 2D polylines in
      that (dir_u, dir_v) basis. sub_anchor_points picks the wall the live
      anchor lies in, then moves the live anchor itself along +/-dir_u and
      +/-dir_v to the first boundary crossing -- no new anchor is
      generated; each sub-anchor differs from the anchor in one in-plane
      coordinate only.

    None when the referenced features resolve to neither shape."""
    cyl: Optional[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = None  # (axis_point, unit axis_dir, boundary pts)
    planes: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []  # (unit normal, dir_u, dir_v, face_id)

    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        for face_id in resolve_face_ids(fn, doc):
            face = doc.faces.get(face_id)
            if face is None:
                continue
            surface = doc.surfaces.get(face.surface_id)
            if isinstance(surface, CylinderSurface):
                pts = _face_boundary_points(face_id, doc)
                if not pts:
                    continue
                norm = np.linalg.norm(surface.axis_dir)
                if norm < 1e-9:
                    continue
                axis_dir = surface.axis_dir / norm
                if cyl is None:
                    cyl = (surface.axis_point, axis_dir, list(pts))
                else:
                    # Only fold in another cylindrical patch when it shares
                    # the *same* infinite axis line -- one physical hole's
                    # wall is commonly split into 2+ B-Rep patches, but a
                    # multi-hole pattern's other holes have parallel-but-
                    # offset axes and must not stretch this one's extent.
                    if abs(abs(float(np.dot(axis_dir, cyl[1]))) - 1.0) < 1e-6 and float(
                        np.linalg.norm(np.cross(cyl[1], surface.axis_point - cyl[0]))
                    ) < 1e-6 * (1.0 + float(np.linalg.norm(cyl[0]))):
                        cyl[2].extend(pts)
            elif isinstance(surface, PlaneSurface):
                normal = np.cross(surface.dir_u, surface.dir_v)
                nnorm = np.linalg.norm(normal)
                du = np.linalg.norm(surface.dir_u)
                dv = np.linalg.norm(surface.dir_v)
                if nnorm < 1e-9 or du < 1e-9 or dv < 1e-9:
                    continue
                normal = normal / nnorm
                # face.turned flips the surface's own parametric normal to
                # the face's actual outward one (same as _face_anchor) --
                # without it, an opposite-parallel-planes pair whose two
                # PlaneSurfaces happen to share a parametrization (e.g. an
                # L-Bracket "1 +/-0.1" whose Z=0 and Z=1 walls both cross-
                # product to +Z) never satisfies the dot < -0.9 pair test
                # below, so the feature silently resolves to no recipe.
                if face.turned:
                    normal = -normal
                planes.append((normal, surface.dir_u / du, surface.dir_v / dv, face_id))

    if cyl is not None:
        axis_point, axis_dir, pts = cyl
        proj = np.asarray([float(np.dot(np.asarray(p) - axis_point, axis_dir)) for p in pts])
        return {
            "kind": "cylinder",
            "axis_point": np.asarray(axis_point, dtype=float),
            "axis_dir": np.asarray(axis_dir, dtype=float),
            "proj_min": float(proj.min()),
            "proj_max": float(proj.max()),
        }

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            if float(np.dot(planes[i][0], planes[j][0])) >= -0.9:
                continue
            _n, dir_u, dir_v, _face_i = planes[i]
            u_axis = np.asarray(dir_u, dtype=float)
            v_axis = np.asarray(dir_v, dtype=float)
            faces = []
            # Both faces of the opposite pair, each projected onto the
            # *shared* (dir_u, dir_v) axes -- they're valid in-plane axes
            # for the second face too, since it's parallel to the first.
            for normal, _du, _dv, face_id in (planes[i], planes[j]):
                surface = doc.surfaces.get(doc.faces[face_id].surface_id)
                origin = np.asarray(surface.origin, dtype=float)
                loops_2d = _face_loops_2d(face_id, doc, origin, u_axis, v_axis)
                if not loops_2d:
                    continue
                faces.append({
                    "origin": origin,
                    "normal": np.asarray(normal, dtype=float),
                    "loops_2d": loops_2d,
                })
            if not faces:
                continue
            return {
                "kind": "planes",
                "axes": [u_axis, v_axis],
                "faces": faces,
            }

    return None


def sub_anchor_points(recipe: Optional[dict], anchor_point: np.ndarray) -> list[np.ndarray]:
    """Turn a resolve_sub_anchor_recipe() result into actual 3D points,
    against `anchor_point` (the annotation's *live* anchor -- see that
    function's own docstring for why this is deferred rather than baked in
    at build time). Every returned point equals `anchor_point` in every
    coordinate except the one being extended:

    * "cylinder" -> 2 points; only the coordinate on axis_dir's dominant
      world axis moves, to each axial end (by the same
      anchor + delta*axis_dir value the extended coordinate would have
      had), so the other two stay exactly equal to the anchor's.
    * "planes" -> 4 points; the live anchor itself is moved along
      -dir_u, +dir_u, -dir_v, +dir_v to the first crossing of the nearest
      wall's trimmed boundary (or the nearest crossing on the opposite
      side when the anchor lies outside the face). No new anchor is
      generated, so the 4 sub-anchors and the anchor are coplanar.

    Empty list when recipe is None."""
    if recipe is None:
        return []
    anchor_point = np.asarray(anchor_point, dtype=float)

    if recipe["kind"] == "cylinder":
        axis_point = recipe["axis_point"]
        axis_dir = recipe["axis_dir"]
        anchor_proj = float(np.dot(anchor_point - axis_point, axis_dir))
        k = int(np.argmax(np.abs(axis_dir)))
        out: list[np.ndarray] = []
        for end_proj in (recipe["proj_min"], recipe["proj_max"]):
            point = anchor_point.copy()
            point[k] = anchor_point[k] + (end_proj - anchor_proj) * float(axis_dir[k])
            out.append(point)
        return out

    if recipe["kind"] == "planes":
        faces = recipe["faces"]
        if not faces:
            return []
        u_axis, v_axis = recipe["axes"]
        axes = (np.asarray(u_axis, dtype=float), np.asarray(v_axis, dtype=float))
        # The wall (of the opposite pair) whose plane the live anchor is
        # nearest to -- only used to pick which face's trimmed boundary to
        # search; no new anchor is derived.
        face = min(faces, key=lambda f: abs(float(np.dot(anchor_point - f["origin"], f["normal"]))))
        origin = np.asarray(face["origin"], dtype=float)
        loops_2d = face["loops_2d"]

        rel = anchor_point - origin
        au0, av0 = float(np.dot(rel, axes[0])), float(np.dot(rel, axes[1]))

        # Each sub-anchor is the live anchor itself, moved along one in-plane
        # axis only (every other coordinate, including the one along the
        # wall normal, stays exactly the anchor's) to the face's boundary:
        # the first crossing of the ray, or -- if the anchor sits outside
        # the face so that ray misses -- the nearest crossing on the
        # opposite side of the same line.
        out: list[np.ndarray] = []
        for axis_idx, sign in ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)):  # -u, +u, -v, +v
            d = (sign, 0.0) if axis_idx == 0 else (0.0, sign)
            t = _ray_polyline_first_t((au0, av0), d, loops_2d)
            if t is None:
                t_back = _ray_polyline_first_t((au0, av0), (-d[0], -d[1]), loops_2d)
                t = -t_back if t_back is not None else 0.0
            out.append(anchor_point + t * axes[axis_idx] * sign)
        return out

    return []


def _point_in_polygon(pt: tuple[float, float], poly: np.ndarray) -> bool:
    """Even-odd ray-cast test: is `pt` inside the closed 2D polyline
    `poly` (Nx2)? Used with the whole-face even-odd fill rule in
    resolve_anchor_candidates -- a lattice point inside an odd number of
    a face's loops is in its actual (trimmed) material, an even number
    (0, or inside both the outer loop and one of its holes) is not --
    so this never needs to know which loop is the outer one."""
    x, y = float(pt[0]), float(pt[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi):
            inside = not inside
        j = i
    return inside


_ANCHOR_CANDIDATE_CORNER_COS = 0.94  # ~20 deg turn -> treated as a real corner
# Two "inset rings" pulled in from the outer boundary toward the face's
# own interior point (fractions of the way in), so there is a populated
# band through the middle of a big face, not only a boundary + a grid.
_ANCHOR_CANDIDATE_INSET_SCALES = (0.62, 0.35)

# User-tunable, persisted by ui/main_window.py (QSettings) and applied via
# set_anchor_candidate_params() before build_annotations() runs.
#   spacing_frac -- candidate pitch as a fraction of the whole-model bbox
#                   diagonal (bigger = fewer, coarser candidates).
#   hard_cap     -- safety ceiling on candidates per characteristic (only
#                   bites on a genuinely large face).
_ANCHOR_CANDIDATE_SPACING_FRAC = 0.065
_ANCHOR_CANDIDATE_HARD_CAP = 220


def set_anchor_candidate_params(
    spacing_frac: Optional[float] = None, hard_cap: Optional[int] = None
) -> None:
    """Override the defaults resolve_anchor_candidates() uses when called
    without explicit ``spacing_frac`` / ``hard_cap`` (which is how
    annotations/builder.py calls it). Takes effect on the next
    build_annotations()."""
    global _ANCHOR_CANDIDATE_SPACING_FRAC, _ANCHOR_CANDIDATE_HARD_CAP
    if spacing_frac is not None and spacing_frac > 0:
        _ANCHOR_CANDIDATE_SPACING_FRAC = float(spacing_frac)
    if hard_cap is not None and hard_cap >= 1:
        _ANCHOR_CANDIDATE_HARD_CAP = int(hard_cap)


def anchor_candidate_params() -> tuple[float, int]:
    """Current (spacing_frac, hard_cap) -- for the UI's initial spinbox
    values."""
    return _ANCHOR_CANDIDATE_SPACING_FRAC, _ANCHOR_CANDIDATE_HARD_CAP


def resolve_anchor_candidates(
    feature_nominal_ids: list[str],
    doc: QifDocument,
    face_anchor: np.ndarray,
    spacing_frac: Optional[float] = None,
    min_spacing_frac: float = 0.02,
    hard_cap: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A finite set of ``(point, unit outward normal)`` positions on the
    B-Rep face(s) a characteristic references -- the only places
    "Balloon" style's automatic anchor re-placement (ui/viewport.py) is
    allowed to move that characteristic's leader anchor to. Camera-
    independent; resolved once at build time (annotations/builder.py) and
    stored on Annotation.anchor_candidates.

    The count SCALES WITH FACE AREA (user 2026-08-31): every point --
    boundary resampling and interior lattice alike -- is laid at an
    ABSOLUTE pitch ``spacing_frac * (whole-model bbox diagonal)``. So a
    small face ends up with only a handful of candidates and a large face
    with many; there is no per-face grid count that would give a tiny
    face the same density as a huge one.

    Two kinds of point, returned in this order:

    * **Edge points** -- every loop (outer and inner) of every referenced
      face: real corners (a polyline vertex whose turn exceeds ~20 deg)
      plus an arc-length resampling of the boundary at the pitch above.
      Corners are never dropped by the safety cap. A cylindrical face's
      "edges" are its two axial-end rings, plus the two angular seams
      when it is only a partial arc.
    * **Interior points** -- a lattice at the same pitch across the
      face's trimmed 2D bounding box (planar) / (axial x angular) extent
      (cylindrical), keeping only points inside the material by the
      whole-face even-odd rule, plus the guaranteed-interior
      _planar_interior_point and (only on a face bigger than a few
      pitches) two inset rings from the outer boundary toward it.

    ``hard_cap`` is only a safety ceiling for the per-frame optimizer --
    a genuinely huge face is thinned back to it (boundary fill first,
    then the interior on a centre-weighted stride). Empty when the
    feature resolves to neither a plane nor a cylinder (e.g. a cone).

    ``spacing_frac`` / ``hard_cap`` default to the module-level values
    (set_anchor_candidate_params, persisted by the UI) when not passed --
    annotations/builder.py calls without them.
    """
    if spacing_frac is None:
        spacing_frac = _ANCHOR_CANDIDATE_SPACING_FRAC
    if hard_cap is None:
        hard_cap = _ANCHOR_CANDIDATE_HARD_CAP

    face_ids: list[str] = []
    for fid in feature_nominal_ids:
        fn = doc.feature_nominals.get(fid)
        if fn is None:
            continue
        for face_id in resolve_face_ids(fn, doc):
            if face_id not in face_ids:
                face_ids.append(face_id)
    if not face_ids:
        return []

    # Absolute length unit = the whole model's bounding-box diagonal (NOT
    # this characteristic's own faces' extent -- that is what used to give
    # a small face the same candidate density as a large one).
    model_diag = 0.0
    vpoints = [v.point for v in doc.vertices.values() if v.point is not None]
    if vpoints:
        varr = np.asarray(vpoints, dtype=float)
        model_diag = float(np.linalg.norm(varr.max(axis=0) - varr.min(axis=0)))
    if model_diag < 1e-9:
        all_boundary: list[np.ndarray] = []
        for face_id in face_ids:
            all_boundary.extend(_face_boundary_points(face_id, doc))
        if all_boundary:
            arr = np.asarray(all_boundary, dtype=float)
            model_diag = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    if model_diag < 1e-9:
        model_diag = 1.0

    cand_spacing = max(spacing_frac * model_diag, 1e-9)
    min_spacing = max(min_spacing_frac * model_diag, 1e-9)
    edge_dedup = 0.4 * min_spacing

    # (point, normal, is_corner) -- is_corner survives any later cap.
    edge_pts: list[tuple[np.ndarray, np.ndarray, bool]] = []
    interior_pts: list[tuple[np.ndarray, np.ndarray]] = []
    # A guaranteed-interior 3D point (last planar face's), used as the
    # centre reference for the inset rings and the interior trim sort.
    interior_centre: Optional[np.ndarray] = None

    def _add_edge(point: np.ndarray, normal: np.ndarray, is_corner: bool) -> None:
        nn = float(np.linalg.norm(normal))
        if nn < 1e-9:
            return
        point = np.asarray(point, dtype=float)
        for idx, (kp, _kn, kc) in enumerate(edge_pts):
            if float(np.linalg.norm(kp - point)) < edge_dedup:
                if is_corner and not kc:
                    edge_pts.pop(idx)
                    break
                return
        edge_pts.append((point, np.asarray(normal, dtype=float) / nn, is_corner))

    def _add_interior(point: np.ndarray, normal: np.ndarray) -> None:
        nn = float(np.linalg.norm(normal))
        if nn < 1e-9:
            return
        point = np.asarray(point, dtype=float)
        for kp, _kn, _kc in edge_pts:
            if float(np.linalg.norm(kp - point)) < min_spacing:
                return
        for kp, _kn in interior_pts:
            if float(np.linalg.norm(kp - point)) < min_spacing:
                return
        interior_pts.append((point, np.asarray(normal, dtype=float) / nn))

    def _resample_loop_2d(poly: np.ndarray, spacing: float, to_3d, normal: np.ndarray) -> None:
        """Every real corner of the closed 2D polyline `poly` (turn over
        ~20 deg), then an arc-length resampling of it at `spacing`."""
        m = len(poly)
        for k in range(m):
            prev_v = poly[k] - poly[(k - 1) % m]
            seg = poly[(k + 1) % m] - poly[k]
            lp = float(np.linalg.norm(prev_v))
            seg_len = float(np.linalg.norm(seg))
            if lp > 1e-9 and seg_len > 1e-9:
                cos_turn = float(np.dot(prev_v / lp, seg / seg_len))
                if cos_turn < _ANCHOR_CANDIDATE_CORNER_COS:
                    _add_edge(to_3d(poly[k]), normal, True)
            if seg_len <= spacing:
                continue
            n_steps = int(seg_len // spacing)
            for s in range(n_steps + 1):
                t = s * spacing / seg_len
                if t >= 1.0:
                    break
                _add_edge(to_3d(poly[k] + t * seg), normal, False)

    for face_id in face_ids:
        face = doc.faces.get(face_id)
        if face is None:
            continue
        surface = doc.surfaces.get(face.surface_id)

        if isinstance(surface, PlaneSurface):
            du_norm = np.linalg.norm(surface.dir_u)
            dv_norm = np.linalg.norm(surface.dir_v)
            if du_norm < 1e-9 or dv_norm < 1e-9:
                continue
            du = surface.dir_u / du_norm
            dv = surface.dir_v / dv_norm
            normal = np.cross(du, dv)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-9:
                continue
            normal = normal / n_norm
            if face.turned:
                normal = -normal
            origin = np.asarray(surface.origin, dtype=float)
            loops_2d = _face_loops_2d(face_id, doc, origin, du, dv)
            if not loops_2d:
                continue
            stacked = np.vstack(loops_2d)
            lo = stacked.min(axis=0)
            hi = stacked.max(axis=0)
            span_u = float(hi[0] - lo[0])
            span_v = float(hi[1] - lo[1])
            face_diag = float(np.hypot(span_u, span_v))

            # (1) boundary -- corners + arc-length resampling at the pitch.
            to_3d = lambda uv: origin + uv[0] * du + uv[1] * dv  # noqa: E731
            for poly in loops_2d:
                _resample_loop_2d(poly, cand_spacing, to_3d, normal)

            # (2) interior -- the guaranteed-interior point, an area-
            #     proportional lattice, and (big faces only) two inset
            #     rings from the outer boundary toward that point.
            interior = _planar_interior_point(face, surface, doc)
            centre_2d = None
            if interior is not None:
                interior = np.asarray(interior, dtype=float)
                interior_centre = interior
                _add_interior(interior, normal)
                rel = interior - origin
                centre_2d = np.array([float(rel @ du), float(rel @ dv)])

            if centre_2d is not None and face_diag > 3.0 * cand_spacing:
                outer = max(
                    loops_2d,
                    key=lambda p: float(
                        (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())
                    ),
                )
                for inset in _ANCHOR_CANDIDATE_INSET_SCALES:
                    m = len(outer)
                    for k in range(m):
                        a2 = outer[k]
                        seg2 = outer[(k + 1) % m] - a2
                        seg2_len = float(np.linalg.norm(seg2))
                        n_steps = max(1, int(seg2_len // cand_spacing))
                        for s in range(n_steps):
                            p2 = a2 + (s / n_steps) * seg2
                            q = centre_2d + inset * (p2 - centre_2d)
                            if sum(1 for poly in loops_2d if _point_in_polygon((q[0], q[1]), poly)) % 2 == 1:
                                _add_interior(origin + q[0] * du + q[1] * dv, normal)

            n_u = max(1, int(round(span_u / cand_spacing)))
            n_v = max(1, int(round(span_v / cand_spacing)))
            for i in range(n_u):
                for j in range(n_v):
                    u = lo[0] + (i + 0.5) / n_u * span_u
                    v = lo[1] + (j + 0.5) / n_v * span_v
                    if sum(1 for poly in loops_2d if _point_in_polygon((u, v), poly)) % 2 == 1:
                        _add_interior(origin + u * du + v * dv, normal)

        elif isinstance(surface, CylinderSurface):
            axis_norm = np.linalg.norm(surface.axis_dir)
            if axis_norm < 1e-9:
                continue
            axis_dir = surface.axis_dir / axis_norm
            pts = _face_boundary_points(face_id, doc)
            if not pts:
                continue
            proj = [float(np.dot(np.asarray(p, dtype=float) - surface.axis_point, axis_dir)) for p in pts]
            proj_min, proj_max = min(proj), max(proj)
            a0, a1 = surface.domain_angle
            radius = surface.diameter / 2
            axial_len = abs(proj_max - proj_min)
            arc_len = abs(a1 - a0) * radius

            def _cyl(axial: float, angle: float, _s=surface, _ad=axis_dir, _r=radius, _f=face):
                radial = rodrigues_rotate(_s.dir_beg, _ad, angle)
                rn = float(np.linalg.norm(radial))
                if rn < 1e-9:
                    return None
                radial = radial / rn
                point = _s.axis_point + axial * _ad + _r * radial
                return point, (-radial if _f.turned else radial)

            # (1) axial-end rings (always) + angular seams (partial arc
            # only) -- resampled at the pitch, so a wide hole gets more
            # ring points than a small one.
            n_ring = max(1, int(round(arc_len / cand_spacing)))
            n_seam = max(1, int(round(axial_len / cand_spacing)))
            for end_proj in (proj_min, proj_max):
                for t in range(n_ring + 1):
                    pn = _cyl(end_proj, a0 + t / n_ring * (a1 - a0))
                    if pn is not None:
                        _add_edge(pn[0], pn[1], False)
            if abs(abs(a1 - a0) - 2 * np.pi) > 1e-3:
                for angle in (a0, a1):
                    for t in range(n_seam + 1):
                        pn = _cyl(proj_min + t / n_seam * (proj_max - proj_min), angle)
                        if pn is not None:
                            _add_edge(pn[0], pn[1], False)

            # (2) interior lattice, area-proportional.
            n_axial = max(1, int(round(axial_len / cand_spacing)))
            n_ang = max(1, int(round(arc_len / cand_spacing)))
            for i in range(n_axial):
                axial = proj_min + (i + 0.5) / n_axial * (proj_max - proj_min)
                for j in range(n_ang):
                    pn = _cyl(axial, a0 + (j + 0.5) / n_ang * (a1 - a0))
                    if pn is not None:
                        _add_interior(pn[0], pn[1])

    # Safety ceiling only. Corners are never trimmed. The boundary fill is
    # capped so it can't dominate; then, if the whole thing is still huge
    # (a genuinely large face), the interior is thinned on a centre-
    # weighted stride.
    corners = [(p, n) for p, n, c in edge_pts if c]
    fills = [(p, n) for p, n, c in edge_pts if not c]
    fill_budget = max(0, min(len(fills), max(len(interior_pts), 8), hard_cap // 3))
    if len(fills) > fill_budget:
        if fill_budget == 0:
            fills = []
        else:
            step = len(fills) / fill_budget
            fills = [fills[int(k * step)] for k in range(fill_budget)]
    edge_final = corners + fills

    interior_budget = max(1, hard_cap - len(edge_final))
    if len(interior_pts) > interior_budget:
        centre = interior_centre if interior_centre is not None else np.asarray(face_anchor, dtype=float)
        interior_pts.sort(key=lambda pn: float(np.linalg.norm(pn[0] - centre)))
        step = len(interior_pts) / interior_budget
        interior_pts = [interior_pts[int(k * step)] for k in range(interior_budget)]

    return edge_final + interior_pts


def resolve_face_ids(feature_nominal: FeatureNominal, doc: QifDocument, _seen: Optional[set] = None) -> list[str]:
    """The B-Rep Face id(s) this feature nominal is built from (QIF's own
    EntityInternalIds), for highlighting "the face(s) this ISO GPS
    characteristic is actually about" on the solid. Composite features
    (kind == "Group") resolve recursively, unioning their sub-features'
    faces."""
    if isinstance(feature_nominal, (PlaneFeatureNominal, CylinderFeatureNominal)):
        return list(feature_nominal.entity_internal_ids)

    if isinstance(feature_nominal, GenericFeatureNominal):
        if feature_nominal.entity_internal_ids:
            return list(feature_nominal.entity_internal_ids)

        if feature_nominal.kind == "Group":
            seen = _seen if _seen is not None else set()
            if feature_nominal.id in seen:
                return []
            seen.add(feature_nominal.id)
            sub_ids = id_list(feature_nominal.raw_element, "FeatureNominalIds")
            face_ids: list[str] = []
            for sub_id in sub_ids:
                sub = doc.feature_nominals.get(sub_id)
                if sub is None:
                    continue
                face_ids.extend(resolve_face_ids(sub, doc, seen))
            return face_ids

        return []  # e.g. OtherSurfaceFeatureNominal without EntityInternalIds

    return []


def count_feature_instances(feature_nominal: FeatureNominal, doc: QifDocument, _seen: Optional[set] = None) -> int:
    """How many distinct physical FEATURE instances this feature nominal
    represents -- unlike resolve_face_ids, a leaf (non-Group) feature
    always counts as exactly 1, no matter how many EntityInternalIds/B-Rep
    faces its own single surface happens to be split across (e.g. one
    through-hole's cylindrical wall commonly resolves to 2 separate B-Rep
    patches, or one V-Block "height" feature is deliberately the pair of
    opposing planes it spans -- neither is "2 features"). Composite
    features (kind == "Group") still resolve recursively, summing their
    sub-features' own counts, exactly like resolve_face_ids.

    Used for a "dimension"-kind characteristic's (Diameter/Radius, not
    Width/Length -- see annotations/builder.py's own count logic) "N×"
    pattern-count notation: 2026-08-22, user-directed bug fix -- an
    L-Bracket Diameter nominal with 4 FeatureNominalIds entries, each one
    hole whose surface happens to resolve to 2 B-Rep faces, was showing
    "8×∅3.5" (len(face_ids)) instead of the correct "4×∅3.5" (one hole per
    entry) -- confirmed against the QIF file's own author-provided
    _3dv.FeatureCounter="4"/_3dv.TextPrefix="4X" attributes, which this
    parser doesn't otherwise read (see parsing/gdt_parser.py's Attributes
    skip)."""
    if isinstance(feature_nominal, GenericFeatureNominal) and feature_nominal.kind == "Group":
        seen = _seen if _seen is not None else set()
        if feature_nominal.id in seen:
            return 0
        seen.add(feature_nominal.id)
        sub_ids = id_list(feature_nominal.raw_element, "FeatureNominalIds")
        total = 0
        for sub_id in sub_ids:
            sub = doc.feature_nominals.get(sub_id)
            if sub is None:
                continue
            total += count_feature_instances(sub, doc, seen)
        return total

    return 1
