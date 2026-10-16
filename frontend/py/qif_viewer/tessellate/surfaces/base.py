from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TriangleSoup:
    points: np.ndarray       # (N, 3) float
    triangles: np.ndarray    # (M, 3) int, indices into points
    normal_hint: Optional[np.ndarray] = None  # informational only
