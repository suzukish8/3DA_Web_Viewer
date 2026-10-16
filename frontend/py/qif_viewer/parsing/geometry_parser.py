"""Parses the QIF B-Rep (Product/GeometrySet + Product/TopologySet) into the
model.brep dataclasses.

Three analytic surface types (Plane23, Cylinder23, Cone23) plus Nurbs23 are
known. Anything else becomes an `UnknownSurface` placeholder so the face
still exists in the model and can fall back to a wireframe at tessellation
time (see tessellate/surfaces/unsupported.py) instead of vanishing.
"""
from __future__ import annotations

import base64
import xml.etree.ElementTree as ET

import numpy as np

from qif_viewer.model.brep import (
    ArcCurve,
    Body,
    CoEdge,
    ConeSurface,
    Curve,
    CylinderSurface,
    Edge,
    EllipseCurve,
    Face,
    LineCurve,
    Loop,
    NurbsCurve,
    NurbsSurface,
    PlaneSurface,
    Surface,
    UnknownCurve,
    UnknownSurface,
    Vertex,
)
from qif_viewer.model.document import QifDocument

from .xml_utils import find, findall, id_list, id_ref, local_tag, parse_float, parse_float_tuple, parse_point, parse_vec, text_of

# id_ref path handling relies on the "<Container><Id>...</Id></Container>" shape
# used throughout QIF for single references (e.g. Face/Surface, Vertex/Point).


def parse_points(geometry_set: ET.Element) -> dict[str, object]:
    point_set = find(geometry_set, "PointSet")
    points: dict[str, object] = {}
    if point_set is None:
        return points
    for point_el in findall(point_set, "Point"):
        pid = point_el.get("id")
        xyz = text_of(point_el, "XYZ")
        if pid is None or xyz is None:
            continue
        points[pid] = parse_vec(xyz)
    return points


def parse_vertices(topology_set: ET.Element, points: dict) -> dict[str, Vertex]:
    vertex_set = find(topology_set, "VertexSet")
    vertices: dict[str, Vertex] = {}
    if vertex_set is None:
        return vertices
    for v_el in findall(vertex_set, "Vertex"):
        vid = v_el.get("id")
        point_id = id_ref(v_el, "Point")
        if vid is None or point_id is None or point_id not in points:
            continue
        vertices[vid] = Vertex(id=vid, point=points[point_id])
    return vertices


def _float_list(el: ET.Element, path: str) -> "np.ndarray | None":
    t = text_of(el, path)
    return np.array(t.split(), dtype=float) if t is not None else None


def _binary_points(el: ET.Element) -> "np.ndarray | None":
    """CPsBinary: base64 little-endian doubles, 3 per control point."""
    t = text_of(el, "CPsBinary")
    if t is None:
        return None
    return np.frombuffer(base64.b64decode(t), dtype="<f8").reshape(-1, 3).copy()


def parse_curves(geometry_set: ET.Element, warnings: list[str]) -> dict[str, Curve]:
    curve_set = find(geometry_set, "Curve13Set")
    curves: dict[str, Curve] = {}
    if curve_set is None:
        return curves
    for el in list(curve_set):
        tag = local_tag(el)
        cid = el.get("id")
        if cid is None:
            continue
        if tag == "Segment13":
            core = find(el, "Segment13Core")
            start = parse_point(core, "StartPoint")
            end = parse_point(core, "EndPoint")
            if start is None or end is None:
                warnings.append(f"Segment13 {cid} missing start/end point; skipped")
                continue
            curves[cid] = LineCurve(id=cid, start=start, end=end)
        elif tag == "ArcCircular13":
            core = find(el, "ArcCircular13Core")
            radius = parse_float(core, "Radius")
            center = parse_point(core, "Center")
            dir_beg = parse_point(core, "DirBeg")
            normal = parse_point(core, "Normal")
            domain_text = core.get("domain") if core is not None else None
            if any(v is None for v in (radius, center, dir_beg, normal, domain_text)):
                warnings.append(f"ArcCircular13 {cid} missing required fields; skipped")
                continue
            domain = parse_float_tuple(domain_text)
            curves[cid] = ArcCurve(id=cid, radius=radius, center=center, dir_beg=dir_beg, normal=normal, domain=domain)
        elif tag == "Nurbs13":
            core = find(el, "Nurbs13Core")
            order = parse_float(core, "Order") if core is not None else None
            knots = _float_list(core, "Knots") if core is not None else None
            ctrl = _binary_points(core) if core is not None else None
            domain_text = core.get("domain") if core is not None else None
            if order is None or knots is None or ctrl is None or domain_text is None                     or len(knots) != len(ctrl) + int(order):
                warnings.append(f"Nurbs13 {cid} malformed; using vertex fallback")
                curves[cid] = UnknownCurve(id=cid, xml_tag=tag)
                continue
            w = _float_list(core, "Weights")
            if w is not None and len(w) != len(ctrl):
                w = None
            curves[cid] = NurbsCurve(
                id=cid, order=int(order), knots=knots, ctrl=ctrl, weights=w,
                domain=parse_float_tuple(domain_text),
            )
        elif tag == "ArcConic13":
            core = find(el, "ArcConic13Core")
            form = core.get("form") if core is not None else None
            if form == "ELLIPSE":
                a = parse_float(core, "A")
                b = parse_float(core, "B")
                center = parse_point(core, "Center")
                dir_beg = parse_point(core, "DirBeg")
                normal = parse_point(core, "Normal")
                domain_text = core.get("domain")
                if any(v is None for v in (a, b, center, dir_beg, normal, domain_text)):
                    warnings.append(f"ArcConic13 {cid} (ELLIPSE) missing required fields; using vertex fallback")
                    curves[cid] = UnknownCurve(id=cid, xml_tag=tag)
                    continue
                curves[cid] = EllipseCurve(
                    id=cid, semi_major=a, semi_minor=b, center=center,
                    dir_beg=dir_beg, normal=normal, domain=parse_float_tuple(domain_text),
                )
            else:
                warnings.append(f"Unsupported ArcConic13 form '{form}' (id={cid}); using vertex fallback")
                curves[cid] = UnknownCurve(id=cid, xml_tag=tag)
        else:
            warnings.append(f"Unsupported curve type '{tag}' (id={cid}); using vertex fallback")
            curves[cid] = UnknownCurve(id=cid, xml_tag=tag)
    return curves


def parse_edges(topology_set: ET.Element, warnings: list[str]) -> dict[str, Edge]:
    edge_set = find(topology_set, "EdgeSet")
    edges: dict[str, Edge] = {}
    if edge_set is None:
        return edges
    for el in findall(edge_set, "Edge"):
        eid = el.get("id")
        curve_id = id_ref(el, "Curve")
        vbeg = id_ref(el, "VertexBeg")
        vend = id_ref(el, "VertexEnd")
        if eid is None or curve_id is None or vbeg is None or vend is None:
            warnings.append(f"Edge {eid} missing curve/vertex reference; skipped")
            continue
        edges[eid] = Edge(id=eid, curve_id=curve_id, vertex_beg_id=vbeg, vertex_end_id=vend)
    return edges


def parse_loops(topology_set: ET.Element, warnings: list[str]) -> dict[str, Loop]:
    loop_set = find(topology_set, "LoopSet")
    loops: dict[str, Loop] = {}
    if loop_set is None:
        return loops
    for el in findall(loop_set, "Loop"):
        lid = el.get("id")
        form = el.get("form", "OUTER")
        coedges: list[CoEdge] = []
        coedges_container = find(el, "CoEdges")
        if coedges_container is not None:
            for ce in findall(coedges_container, "CoEdge"):
                oriented = find(ce, "EdgeOriented")
                if oriented is None:
                    continue
                edge_id = text_of(oriented, "Id")
                turned = oriented.get("turned", "0") == "1"
                if edge_id is None:
                    continue
                coedges.append(CoEdge(edge_id=edge_id, turned=turned))
        if lid is None:
            continue
        loops[lid] = Loop(id=lid, form=form, coedges=coedges)
    return loops


_SURFACE_TAGS = {"Plane23", "Cylinder23", "Cone23"}


def parse_surfaces(geometry_set: ET.Element, warnings: list[str]) -> dict[str, Surface]:
    surface_set = find(geometry_set, "SurfaceSet")
    surfaces: dict[str, Surface] = {}
    if surface_set is None:
        return surfaces
    for el in list(surface_set):
        tag = local_tag(el)
        sid = el.get("id")
        if sid is None:
            continue
        if tag == "Plane23":
            core = find(el, "Plane23Core")
            origin = parse_point(core, "Origin")
            dir_u = parse_point(core, "DirU")
            dir_v = parse_point(core, "DirV")
            if any(v is None for v in (origin, dir_u, dir_v)):
                warnings.append(f"Plane23 {sid} missing core fields; skipped")
                continue
            surfaces[sid] = PlaneSurface(id=sid, origin=origin, dir_u=dir_u, dir_v=dir_v)
        elif tag == "Cylinder23":
            core = find(el, "Cylinder23Core")
            diameter = parse_float(core, "Diameter")
            length = parse_float(core, "Length")
            axis = find(core, "Axis")
            sweep = find(core, "Sweep")
            axis_point = parse_point(axis, "AxisPoint") if axis is not None else None
            axis_dir = parse_point(axis, "Direction") if axis is not None else None
            dir_beg = parse_point(sweep, "DirBeg") if sweep is not None else None
            domain_text = text_of(sweep, "DomainAngle") if sweep is not None else None
            if any(v is None for v in (diameter, length, axis_point, axis_dir, dir_beg, domain_text)):
                warnings.append(f"Cylinder23 {sid} missing core fields; skipped")
                continue
            surfaces[sid] = CylinderSurface(
                id=sid, diameter=diameter, length=length,
                axis_point=axis_point, axis_dir=axis_dir,
                dir_beg=dir_beg, domain_angle=parse_float_tuple(domain_text),
            )
        elif tag == "Cone23":
            core = find(el, "Cone23Core")
            d_bottom = parse_float(core, "DiameterBottom")
            d_top = parse_float(core, "DiameterTop")
            length = parse_float(core, "Length")
            axis = find(core, "Axis")
            sweep = find(core, "Sweep")
            axis_point = parse_point(axis, "AxisPoint") if axis is not None else None
            axis_dir = parse_point(axis, "Direction") if axis is not None else None
            dir_beg = parse_point(sweep, "DirBeg") if sweep is not None else None
            domain_text = text_of(sweep, "DomainAngle") if sweep is not None else None
            if any(v is None for v in (d_bottom, d_top, length, axis_point, axis_dir, dir_beg, domain_text)):
                warnings.append(f"Cone23 {sid} missing core fields; skipped")
                continue
            surfaces[sid] = ConeSurface(
                id=sid, diameter_bottom=d_bottom, diameter_top=d_top, length=length,
                axis_point=axis_point, axis_dir=axis_dir,
                dir_beg=dir_beg, domain_angle=parse_float_tuple(domain_text),
            )
        elif tag == "Nurbs23":
            core = find(el, "Nurbs23Core")
            ou = parse_float(core, "OrderU") if core is not None else None
            ov = parse_float(core, "OrderV") if core is not None else None
            ku = _float_list(core, "KnotsU") if core is not None else None
            kv = _float_list(core, "KnotsV") if core is not None else None
            cps = _binary_points(core) if core is not None else None
            ok = all(x is not None for x in (ou, ov, ku, kv, cps))
            if ok:
                nu = len(ku) - int(ou)
                nv = len(kv) - int(ov)
                ok = nu > 0 and nv > 0 and nu * nv == len(cps)
            if not ok:
                surfaces[sid] = UnknownSurface(id=sid, xml_tag=tag)
                warnings.append(f"Nurbs23 {sid} malformed; will render as fallback")
                continue
            # Control points are stored U-fastest: index = j*nu + i.
            ctrl = cps.reshape(nv, nu, 3).transpose(1, 0, 2).copy()
            w = _float_list(core, "Weights")
            if w is not None:
                w = w.reshape(nv, nu).T.copy() if len(w) == nu * nv else None
            surfaces[sid] = NurbsSurface(
                id=sid, order_u=int(ou), order_v=int(ov), knots_u=ku, knots_v=kv,
                ctrl=ctrl, weights=w,
            )
        else:
            surfaces[sid] = UnknownSurface(id=sid, xml_tag=tag)
            warnings.append(f"Unsupported surface type '{tag}' (id={sid}); will render as fallback")
    return surfaces


def _parse_color(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.6, 0.6, 0.6)
    parts = [float(x) for x in text.split()]
    # QIF colors are 0-255 ints in this sample; normalize to 0-1 for VTK.
    return tuple(p / 255.0 if p > 1.0 else p for p in parts[:3])


def parse_faces(topology_set: ET.Element, warnings: list[str]) -> dict[str, Face]:
    face_set = find(topology_set, "FaceSet")
    faces: dict[str, Face] = {}
    if face_set is None:
        return faces
    for el in findall(face_set, "Face"):
        fid = el.get("id")
        surface_id = id_ref(el, "Surface")
        loop_ids = id_list(el, "LoopIds")
        if fid is None or surface_id is None:
            warnings.append(f"Face {fid} missing surface reference; skipped")
            continue
        faces[fid] = Face(
            id=fid,
            label=el.get("label", ""),
            color=_parse_color(el.get("color")),
            turned=el.get("turned", "0") == "1",
            surface_id=surface_id,
            loop_ids=loop_ids,
        )
    return faces


def parse_bodies(topology_set: ET.Element) -> dict[str, Body]:
    body_set = find(topology_set, "BodySet")
    bodies: dict[str, Body] = {}
    if body_set is None:
        return bodies
    for el in findall(body_set, "Body"):
        bid = el.get("id")
        if bid is None:
            continue
        bodies[bid] = Body(
            id=bid,
            face_ids=id_list(el, "FaceIds"),
            loop_ids=id_list(el, "LoopIds"),
            edge_ids=id_list(el, "EdgeIds"),
            vertex_ids=id_list(el, "VertexIds"),
        )
    return bodies


def parse_geometry(root: ET.Element, doc: QifDocument) -> None:
    product = find(root, "Product")
    if product is None:
        doc.warn("No <Product> element found; document has no geometry")
        return
    geometry_set = find(product, "GeometrySet")
    topology_set = find(product, "TopologySet")
    if geometry_set is None or topology_set is None:
        doc.warn("Missing GeometrySet/TopologySet under <Product>; no geometry parsed")
        return

    points = parse_points(geometry_set)
    doc.vertices = parse_vertices(topology_set, points)
    doc.curves = parse_curves(geometry_set, doc.warnings)
    doc.edges = parse_edges(topology_set, doc.warnings)
    doc.loops = parse_loops(topology_set, doc.warnings)
    doc.surfaces = parse_surfaces(geometry_set, doc.warnings)
    doc.faces = parse_faces(topology_set, doc.warnings)
    doc.bodies = parse_bodies(topology_set)
