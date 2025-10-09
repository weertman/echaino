# FILE: core\measurement_engine.py
# PATH: D:\urchinScanner\core\measurement_engine.py

import numpy as np
from typing import List, Optional, Tuple
from data.models import Segmentation
import cv2
from scipy.optimize import least_squares  # For circle fitting
from shapely.geometry import Polygon  # For fallbacks
import logging

logger = logging.getLogger(__name__)


class MeasurementEngine:
    """Calculate real-world measurements from segmentations"""

    def __init__(self, scale_px_per_cm: Optional[float] = None):
        self.scale_px_per_cm = scale_px_per_cm

    def update_scale(self, scale_px_per_cm: float):
        """Update the scale factor"""
        self.scale_px_per_cm = scale_px_per_cm

    def calculate_measurements(self, segmentation: Segmentation) -> Segmentation:
        """Calculate all measurements for a segmentation with polygon fallbacks"""

        # NEW: Fallback to polygon if mask is None and polygon exists
        if segmentation.mask is None and segmentation.polygon:
            try:
                poly = Polygon(segmentation.polygon)
                if poly.is_valid and len(segmentation.polygon) >= 3:
                    # Centroid from Shapely
                    cent = poly.centroid
                    segmentation.centroid = (cent.x, cent.y)

                    # Perimeter from cv2.arcLength on polygon points (closed=True)
                    pts = np.array(segmentation.polygon, dtype=np.int32)
                    segmentation.perimeter_pixels = cv2.arcLength(pts, True)  # Confirmed via search

                    # Area from Shapely (vector-based approx, close to raster)
                    if segmentation.area_pixels == 0:  # Only override if not precomputed
                        segmentation.area_pixels = poly.area
                else:
                    logger.warning(f"Invalid polygon for seg {segmentation.id}; skipping fallbacks")
            except Exception as e:
                logger.error(f"Polygon fallback failed for seg {segmentation.id}: {e}")

        # Existing: If mask present, use it (overrides fallbacks if needed)
        elif segmentation.mask is not None:
            segmentation.centroid = self._calculate_centroid(segmentation.mask)
            segmentation.perimeter_pixels = self._calculate_perimeter(segmentation.mask)

        # Existing scaling if scale available (now works with precomputed/fallback values)
        if self.scale_px_per_cm:
            segmentation.area_cm2 = segmentation.area_pixels / (self.scale_px_per_cm ** 2)
            segmentation.perimeter_cm = segmentation.perimeter_pixels / self.scale_px_per_cm

            # NEW: Compute diameter_um if not set (moved here for consistency; uses existing method)
            if segmentation.diameter_um is None:
                segmentation.diameter_um = self._calculate_best_fit_diameter(segmentation.polygon)

        return segmentation

    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Calculate mask centroid"""
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            return (0.0, 0.0)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy)

    def _calculate_perimeter(self, mask: np.ndarray) -> float:
        """Calculate mask perimeter"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        return cv2.arcLength(contours[0], True)

    def _calculate_best_fit_diameter(self, polygon: Optional[List[List[float]]]) -> Optional[float]:
        """Calculate diameter of best-fit circle to polygon points in micrometers"""
        if not polygon or len(polygon) < 3 or self.scale_px_per_cm is None:
            return None

        points = np.array(polygon, dtype=np.float64)

        def calc_R(xc, yc):
            return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

        def f_2(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = np.mean(points, axis=0)
        result = least_squares(f_2, center_estimate)
        if not result.success:
            return None

        center = result.x
        Ri = calc_R(*center)
        r_pixels = Ri.mean()

        # Convert to micrometers: diameter_pixels = 2 * r_pixels
        # pixels per µm = scale_px_per_cm / 10000 (since 1 cm = 10000 µm)
        # diameter_um = diameter_pixels / (pixels per µm) = diameter_pixels * (10000 / scale_px_per_cm)
        diameter_pixels = 2 * r_pixels
        diameter_um = diameter_pixels * (10000 / self.scale_px_per_cm)

        return diameter_um

    def calculate_diameter_um(self, polygon: List[List[float]]) -> Optional[float]:
        """Public method to calculate diameter in micrometers from polygon (for on-demand computation)"""
        return self._calculate_best_fit_diameter(polygon)

    def batch_calculate(self, segmentations: List[Segmentation]) -> List[Segmentation]:
        """Calculate measurements for multiple segmentations"""
        return [self.calculate_measurements(seg) for seg in segmentations]