"""ブラウザ版用の最小スタブ。qif_viewer.tessellate.mesh_builder が使う PolyData だけを提供する。"""
import numpy as np


class PolyData:
    def __init__(self, points=None, faces=None):
        self.points = np.zeros((0, 3)) if points is None else np.asarray(points, dtype=float)
        self.faces = np.zeros(0, dtype=np.int64) if faces is None else np.asarray(faces)
        self.cell_data = {}

    @property
    def n_points(self):
        return len(self.points)

    @property
    def bounds(self):
        p = self.points
        if len(p) == 0:
            return (0, 0, 0, 0, 0, 0)
        mn, mx = p.min(axis=0), p.max(axis=0)
        return (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
