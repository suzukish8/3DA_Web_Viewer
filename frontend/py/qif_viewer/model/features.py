"""Feature nominals: semantic anchors used to place ISO GPS annotations."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class PlaneFeatureNominal:
    id: str
    name: str
    feature_definition_id: str
    location: np.ndarray
    normal: np.ndarray
    # B-Rep Face ids this nominal feature is built from (QIF's own
    # EntityInternalIds) -- lets the UI highlight "the face(s) this ISO GPS
    # characteristic is actually about" on the solid.
    entity_internal_ids: list[str] = field(default_factory=list)


@dataclass
class CylinderFeatureNominal:
    id: str
    name: str
    feature_definition_id: str
    axis_point: np.ndarray
    axis_dir: np.ndarray
    entity_internal_ids: list[str] = field(default_factory=list)


@dataclass
class GenericFeatureNominal:
    """Best-effort catch-all for Group/OppositeParallelPlanes/OtherSurface/etc.

    `raw_element` is an intentional escape hatch so unusual feature types can
    still be inspected later without the parser needing to model every field.
    """
    id: str
    name: str
    kind: str
    location: Optional[np.ndarray]
    axis_dir: Optional[np.ndarray]
    raw_element: ET.Element
    entity_internal_ids: list[str] = field(default_factory=list)


FeatureNominal = Union[PlaneFeatureNominal, CylinderFeatureNominal, GenericFeatureNominal]
