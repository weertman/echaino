# FILE: data\models.py
# PATH: D:\urchinScanner\data\models.py

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np  # Added for mask type


@dataclass
class BoundingBox:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    status: str = "pending"  # Added for status tracking
    source: Optional[str] = 'manual'  # New field: tracks origin ('manual' or 'yolo')
    class_id: int = 0  # NEW: Multi-class support, default 0 for single-class compat

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if boxes intersect"""
        return not (self.x2 < other.x1 or self.x1 > other.x2 or
                    self.y2 < other.y1 or self.y1 > other.y2)

    # Add a helper method for source setting (optional, for convenience)
    def set_source(self, source: str):
        self.source = source


@dataclass
class Segmentation:
    id: int
    box_id: int
    mask: Optional[np.ndarray] = None
    polygon: Optional[List[List[float]]] = None
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h
    area_pixels: float = 0.0
    confidence: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    perimeter_pixels: float = 0.0
    area_cm2: Optional[float] = None
    perimeter_cm: Optional[float] = None
    source: Optional[str] = 'manual'  # New field: matches box source for consistency
    class_id: int = 0  # NEW: Multi-class support, default 0
    class_name: Optional[str] = None  # NEW: Human-readable name (from config['classes'])
    diameter_um: Optional[float] = None  # NEW: Diameter of best-fit circle in micrometers

    # Add a helper method similar to BoundingBox
    def set_source(self, source: str):
        self.source = source


@dataclass
class Tile:
    id: int
    x: int
    y: int
    width: int
    height: int
    boxes: List[BoundingBox] = field(default_factory=list)
    segmentations: List[Segmentation] = field(default_factory=list)  # NEW: Added for training with segmentations

    def clip_box(self, box: BoundingBox) -> Optional[BoundingBox]:
        """Clip box to tile boundaries"""
        x1 = max(self.x, box.x1)
        y1 = max(self.y, box.y1)
        x2 = min(self.x + self.width, box.x2)
        y2 = min(self.y + self.height, box.y2)

        if x1 < x2 and y1 < y2:
            return BoundingBox(box.id, x1, y1, x2, y2)
        return None

# In data\models.py, add after Segmentation:
@dataclass
class CalibrationLine:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    real_distance: float
    unit: str = "cm"

    @property
    def pixel_length(self) -> float:
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        return (dx ** 2 + dy ** 2) ** 0.5

    @property
    def pixels_per_unit(self) -> float:
        return self.pixel_length / self.real_distance if self.real_distance > 0 else 0.0


@dataclass
class ProjectData:
    image_path: str
    image_shape: Tuple[int, int, int]
    boxes: List[BoundingBox] = field(default_factory=list)
    all_boxes: List[BoundingBox] = field(default_factory=list)  # Added
    segmentations: List[Segmentation] = field(default_factory=list)
    scale_px_per_cm: Optional[float] = None
    calibration_line: Optional[CalibrationLine] = None
    metadata: Dict = field(default_factory=lambda: {'classes': []})  # NEW: Default with empty classes list
    archive_path: Optional[str] = None  # Added