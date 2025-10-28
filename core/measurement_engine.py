# FILE: core/measurement_engine.py
# PATH: D:\\echaino\\core\\measurement_engine.py

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares  # For circle fitting
from shapely.geometry import Polygon      # Used for polygon fallbacks / area

from data.models import Segmentation

logger = logging.getLogger(__name__)


class MeasurementEngine:
    """
    Compute real-world measurements from segmentations.

    Key behaviors:
      - Works with either a binary mask or a polygon.
      - Converts pixel-based metrics to cm/µm when scale (px/cm) is available.
      - Computes BOTH:
          * diameter_best_fit_um  : best-fit circle to the polygon
          * diameter_feret_max_um : Feret diameter (maximum caliper / tip-to-tip)
        ...and keeps the legacy alias:
          * diameter_um           : equals diameter_best_fit_um for compatibility
    """

    def __init__(self, scale_px_per_cm: Optional[float] = None):
        self.scale_px_per_cm = scale_px_per_cm

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def update_scale(self, scale_px_per_cm: float) -> None:
        """Update the scale factor (pixels per centimeter)."""
        self.scale_px_per_cm = float(scale_px_per_cm) if scale_px_per_cm is not None else None

    def calculate_measurements(self, segmentation: Segmentation) -> Segmentation:
        """
        Calculate measurements for a single segmentation.

        Fills (if available from inputs):
          - centroid (pixels)
          - perimeter_pixels
          - area_pixels

        And, if scale is known (px/cm), also:
          - area_cm2
          - perimeter_cm
          - diameter_best_fit_um
          - diameter_feret_max_um
          - diameter_um  (legacy alias = best-fit)
        """

        # --------------------------
        # 1) Pixel-domain basics
        # --------------------------
        # Prefer mask when available; otherwise, fall back to polygon-only computations.
        if segmentation.mask is not None:
            # From mask
            try:
                segmentation.centroid = self._calculate_centroid(segmentation.mask)
            except Exception as e:
                logger.warning(f"Centroid from mask failed: {e}")
                segmentation.centroid = segmentation.centroid or (0.0, 0.0)

            try:
                segmentation.perimeter_pixels = self._calculate_perimeter(segmentation.mask)
            except Exception as e:
                logger.warning(f"Perimeter from mask failed: {e}")
                segmentation.perimeter_pixels = segmentation.perimeter_pixels or 0.0

            # area_pixels should already be set upstream; keep if present
            if not segmentation.area_pixels:
                try:
                    segmentation.area_pixels = float(np.sum(segmentation.mask.astype(bool)))
                except Exception as e:
                    logger.warning(f"Area from mask failed: {e}")
                    segmentation.area_pixels = segmentation.area_pixels or 0.0

        else:
            # From polygon fallback (no mask)
            if segmentation.polygon:
                try:
                    poly = Polygon(segmentation.polygon)
                    if poly.is_valid and len(segmentation.polygon) >= 3:
                        # Centroid from shapely
                        c = poly.centroid
                        segmentation.centroid = (c.x, c.y)

                        # Perimeter from polygon using cv2 (faster than shapely length here)
                        pts = np.array(segmentation.polygon, dtype=np.int32)
                        segmentation.perimeter_pixels = float(cv2.arcLength(pts, True))

                        # Area (pixels) (use polygon area if not already set)
                        if not segmentation.area_pixels:
                            segmentation.area_pixels = float(poly.area)
                    else:
                        logger.warning(f"Invalid polygon for seg {segmentation.id}; fallback skipped")
                except Exception as e:
                    logger.error(f"Polygon fallback failed for seg {segmentation.id}: {e}")
            else:
                logger.debug(f"Seg {segmentation.id}: neither mask nor polygon available for pixel metrics.")

        # --------------------------
        # 2) Scaled metrics (requires px/cm)
        # --------------------------
        if self.scale_px_per_cm:
            try:
                # Basic conversions
                if segmentation.area_pixels is not None:
                    segmentation.area_cm2 = segmentation.area_pixels / (self.scale_px_per_cm ** 2)
                if segmentation.perimeter_pixels is not None:
                    segmentation.perimeter_cm = segmentation.perimeter_pixels / self.scale_px_per_cm

                # --- Best-fit circle diameter (µm) ---
                if getattr(segmentation, "diameter_best_fit_um", None) is None:
                    bf = self._calculate_best_fit_diameter(segmentation.polygon)
                    segmentation.diameter_best_fit_um = bf

                # --- Feret max (tip-to-tip) diameter (µm) ---
                if getattr(segmentation, "diameter_feret_max_um", None) is None:
                    fm = self._calculate_feret_diameter_max(segmentation.polygon)
                    segmentation.diameter_feret_max_um = fm

            except Exception as e:
                logger.error(f"Scaled metric computation failed for seg {segmentation.id}: {e}")

        return segmentation

    def batch_calculate(self, segmentations: List[Segmentation]) -> List[Segmentation]:
        """Calculate measurements for multiple segmentations."""
        return [self.calculate_measurements(seg) for seg in segmentations]

    # --------------------------
    # Public convenience aliases
    # --------------------------

    def calculate_diameter_um(self, polygon: List[List[float]]) -> Optional[float]:
        """Legacy: best-fit diameter in µm (alias of best-fit)."""
        return self._calculate_best_fit_diameter(polygon)

    def calculate_best_fit_diameter_um(self, polygon: List[List[float]]) -> Optional[float]:
        """Explicit best-fit diameter in µm."""
        return self._calculate_best_fit_diameter(polygon)

    def calculate_feret_diameter_um(self, polygon: List[List[float]]) -> Optional[float]:
        """Explicit Feret (max) diameter in µm."""
        return self._calculate_feret_diameter_max(polygon)

    # ---------------------------------------------------------------------
    # Internal helpers (pixel domain)
    # ---------------------------------------------------------------------

    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Centroid from a binary mask (pixel coordinates)."""
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            return (0.0, 0.0)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy)

    def _calculate_perimeter(self, mask: np.ndarray) -> float:
        """Perimeter length in pixels from a binary mask."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        return float(cv2.arcLength(contours[0], True))

    # ---------------------------------------------------------------------
    # Internal helpers (diameters in µm; require scale_px_per_cm)
    # ---------------------------------------------------------------------

    def _calculate_best_fit_diameter(self, polygon: Optional[List[List[float]]]) -> Optional[float]:
        """
        Best-fit circle diameter (µm) to polygon vertices.
        Returns None if polygon is invalid or scale is unknown.
        """
        if not polygon or len(polygon) < 3 or self.scale_px_per_cm is None:
            return None

        pts = np.asarray(polygon, dtype=np.float64)

        def calc_R(xc, yc):
            return np.sqrt((pts[:, 0] - xc) ** 2 + (pts[:, 1] - yc) ** 2)

        def residuals(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = np.mean(pts, axis=0)
        result = least_squares(residuals, center_estimate, method='lm')
        if not result.success:
            logger.debug("Least-squares circle fit did not converge.")
            return None

        center = result.x
        Ri = calc_R(*center)
        r_pixels = float(Ri.mean())

        # px → µm : diameter_px * (10000 / px_per_cm)
        diameter_px = 2.0 * r_pixels
        return diameter_px * (10000.0 / self.scale_px_per_cm)

    def _calculate_feret_diameter_max(self, polygon: Optional[List[List[float]]]) -> Optional[float]:
        """
        Feret (max caliper / tip-to-tip) diameter (µm).
        Uses convex hull vertices and O(n^2) pairwise scan; fast enough for typical polygons.
        """
        if not polygon or len(polygon) < 2 or self.scale_px_per_cm is None:
            return None

        pts = np.asarray(polygon, dtype=np.float64)
        if pts.shape[0] < 2:
            return None

        try:
            hull = cv2.convexHull(pts, returnPoints=True).reshape(-1, 2)
        except Exception as e:
            logger.warning(f"convexHull failed; falling back to raw points: {e}")
            hull = pts

        if hull.shape[0] < 2:
            return None

        maxd2 = 0.0
        # Simple O(n^2) scan over hull vertices
        for i in range(len(hull)):
            diffs = hull[i + 1:] - hull[i]
            if diffs.size:
                d2 = (diffs ** 2).sum(axis=1).max()
                if d2 > maxd2:
                    maxd2 = float(d2)

        diameter_px = float(np.sqrt(maxd2))
        return diameter_px * (10000.0 / self.scale_px_per_cm)
