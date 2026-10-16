"""Feasibility check: read STEP AP242 shape + PMI via OCP (OCCT Python bindings)."""
import sys

from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Controller
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.IFSelect import IFSelect_RetDone

path = sys.argv[1] if len(sys.argv) > 1 else r""

STEPCAFControl_Controller.Init_s()

app = XCAFApp_Application.GetApplication_s()
doc = TDocStd_Document(TCollection_ExtendedString("mdtv-xcaf"))
app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

reader = STEPCAFControl_Reader()
reader.SetColorMode(True)
reader.SetNameMode(True)
reader.SetLayerMode(True)
# PMI / GD&T
try:
    reader.SetGDTMode(True)
    reader.SetViewMode(True)
    reader.SetPropsMode(True)
    reader.SetMetaMode(True)
    print("GDTMode set OK")
except Exception as e:
    print("GDTMode not available:", e)

status = reader.ReadFile(path)
print("ReadFile status:", status, "OK?" , status == IFSelect_RetDone)

ok = reader.Transfer(doc)
print("Transfer OK?", ok)

shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
labels = TDF_LabelSequence()
shape_tool.GetFreeShapes(labels)
print("Free shape labels:", labels.Length())

total_faces = 0
total_tris = 0
for i in range(1, labels.Length() + 1):
    label = labels.Value(i)
    shape = shape_tool.GetShape_s(label)
    BRepMesh_IncrementalMesh(shape, 0.1)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            total_faces += 1
            total_tris += tri.NbTriangles()
        exp.Next()

print("Total faces with mesh:", total_faces, "Total triangles:", total_tris)

# PMI / GD&T via XCAFDoc_DimTolTool
try:
    from OCP.XCAFDoc import XCAFDoc_DimTolTool
    dimtol_tool = XCAFDoc_DimTolTool.Set_s(doc.Main())
    for name in ("GetDimTolLabels", "GetDimensionLabels", "GetGeomToleranceLabels", "GetDatumLabels"):
        seq = TDF_LabelSequence()
        getattr(dimtol_tool, name)(seq)
        print(f"{name}:", seq.Length())
    pres = TDF_LabelSequence()
    dimtol_tool.GetGDTPresentations(pres)
    print("GetGDTPresentations:", pres.Length())
except Exception as e:
    print("DimTolTool error:", e)
