# FILE: utils/geometry_utils.py
# PATH: D:\echaino\utils\geometry_utils.py

import numpy as np
import cv2
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely import union_all  # Shapely 2.0+
import logging

from data.models import Segmentation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def normalize_mask_uint8(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Normalize any mask-like array to uint8 with values in {0,255}.
    Returns None if input is None.
    """
    if mask is None:
        return None
    if mask.dtype == np.uint8:
        # Map any nonzero to 255 to be explicit
        return (mask > 0).astype(np.uint8) * 255
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    # float or int types
    return (mask > 0).astype(np.uint8) * 255


def _largest_contour(mask_u8: np.ndarray):
    """
    Find the largest external contour on a uint8 mask in {0,255}.
    Returns (contour, area). Contour has shape (N,1,2) as OpenCV returns.
    Returns (None, 0.0) if none found.
    """
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    return c, float(cv2.contourArea(c))


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------

def mask_to_polygon(mask: Optional[np.ndarray], simplify: bool = True) -> List[List[float]]:
    """
    Convert a (binary-ish) mask to a polygon by taking the largest external contour.
    Returns list of [x, y] floats. Returns [] if no valid polygon can be extracted.

    Robustness improvements:
      - Normalizes mask to uint8 {0,255}
      - Selects largest external contour
      - Optional Douglas-Peucker simplification
      - Ensures >= 3 points and positive area
    """
    if mask is None:
        logger.warning("mask_to_polygon: None mask")
        return []

    try:
        mask_u8 = normalize_mask_uint8(mask)
        if mask_u8 is None or not np.any(mask_u8):
            logger.debug("mask_to_polygon: empty mask after normalization")
            return []

        contour, area = _largest_contour(mask_u8)
        if contour is None:
            logger.debug("mask_to_polygon: no contours found")
            return []
        if area < 10.0:  # minimum area threshold in px
            logger.debug(f"mask_to_polygon: contour too small (area={area:.2f})")
            return []

        if simplify:
            # Epsilon proportional to perimeter
            peri = cv2.arcLength(contour, True)
            eps = 0.01 * peri  # tuned for stability; lower = more detail
            contour = cv2.approxPolyDP(contour, eps, True)

        # Flatten to (N,2) and cast to float32
        pts = contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            logger.debug(f"mask_to_polygon: too few points after simplify (n={pts.shape[0]})")
            return []

        # Return as list of [x,y] floats
        return [[float(x), float(y)] for x, y in pts]

    except Exception as e:
        logger.error(f"mask_to_polygon: error converting mask to polygon: {e}")
        import traceback
        traceback.print_exc()
        return []


def calculate_bbox(mask: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from mask (xmin, ymin, width, height).

    Works for any dtype; internally binarizes.
    """
    if mask is None:
        return (0.0, 0.0, 0.0, 0.0)

    binmask = (normalize_mask_uint8(mask) > 0)
    if not np.any(binmask):
        return (0.0, 0.0, 0.0, 0.0)

    rows = np.any(binmask, axis=1)
    cols = np.any(binmask, axis=0)

    if not rows.any() or not cols.any():
        return (0.0, 0.0, 0.0, 0.0)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # +1 to convert from inclusive indices to width/height
    return (float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1))


def calculate_area(mask: Optional[np.ndarray]) -> float:
    """
    Calculate area of mask in pixels.

    Correct for any dtype (including uint8 {0,255}); counts foreground pixels, not sums intensity.
    """
    if mask is None:
        return 0.0
    binmask = (normalize_mask_uint8(mask) > 0)
    return float(np.count_nonzero(binmask))


def polygon_to_mask(polygon: List[List[float]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert polygon to binary mask of given (H, W) shape.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon:
        pts = np.array(polygon, dtype=np.int32)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


def simplify_polygon(polygon: List[List[float]], tolerance: float = 1.0) -> List[List[float]]:
    """
    Simplify polygon using Douglas-Peucker via Shapely, preserving topology.
    Returns a simplified polygon (list of [x,y]) or the original on failure.
    """
    try:
        poly = Polygon(polygon)
        simplified = poly.simplify(tolerance, preserve_topology=True)
        coords = list(simplified.exterior.coords[:-1])  # Remove duplicate last point
        if len(coords) < 3:
            return polygon
        return [[float(x), float(y)] for x, y in coords]
    except Exception as e:
        logger.error(f"simplify_polygon: {e}")
        return polygon


def calculate_polygon_iou(polygon1: List[List[float]], polygon2: List[List[float]]) -> float:
    """
    Calculate IoU between two polygons using Shapely.
    """
    try:
        poly1 = Polygon(polygon1)
        poly2 = Polygon(polygon2)

        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        inter = poly1.intersection(poly2).area
        uni = poly1.union(poly2).area
        return inter / uni if uni > 0 else 0.0
    except Exception as e:
        logger.warning(f"calculate_polygon_iou: {e}")
        return 0.0


def union_polygons(polygons: List[List[List[float]]]) -> List[List[float]]:
    """
    Union multiple polygons using Shapely, handling MultiPolygon and artifacts.
    Returns a single outer polygon (largest area) or [] on failure.
    """
    try:
        if not polygons:
            return []

        shapely_polys = []
        for poly in polygons:
            if not poly or len(poly) < 3:
                continue
            p = Polygon(poly)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_valid and p.area > 0:
                shapely_polys.append(p)

        if not shapely_polys:
            return []

        unioned = union_all(shapely_polys)

        if isinstance(unioned, MultiPolygon):
            unioned = max(unioned.geoms, key=lambda g: g.area)

        # mild simplify to remove tiny artifacts
        unioned = unioned.simplify(1.0, preserve_topology=True)

        if not unioned.is_valid or unioned.area < 10.0:
            logger.debug("union_polygons: union invalid or tiny")
            return []

        coords = list(unioned.exterior.coords[:-1])
        if len(coords) < 3:
            return []
        return [[float(x), float(y)] for x, y in coords]

    except Exception as e:
        logger.error(f"union_polygons: {e}")
        import traceback
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------
# Conversions / SAHI
# ---------------------------------------------------------------------

def polygon_to_yolo_format(polygon: List[List[float]], img_w: int, img_h: int, class_id: int = 0) -> str:
    """
    Convert polygon to YOLO segmentation label string.
    """
    normalized = [f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in polygon]
    return f"{class_id} {' '.join(normalized)}"


def sahi_result_to_segmentation(obj) -> Segmentation:
    """
    Convert a SAHI prediction result to our Segmentation model.
    Expects obj.bbox.to_xywh(), obj.mask.bool_mask (optional), and obj.score.value.
    """
    import logging
    logger = logging.getLogger(__name__)

    bbox = obj.bbox.to_xywh()  # (x, y, w, h)
    mask_np = obj.mask.bool_mask if getattr(obj, 'mask', None) is not None else None

    if mask_np is None:
        logger.debug("sahi_result_to_segmentation: no mask; using bbox approximations")
        return Segmentation(
            id=0,
            box_id=0,
            mask=None,
            polygon=[],
            bbox=bbox,
            area_pixels=0.0,
            confidence=obj.score.value,
            centroid=(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
            perimeter_pixels=0.0,
            source='yolo',
            class_id=getattr(getattr(obj, 'category', None), 'id', 0)
        )

    # Normalize once
    mask_u8 = normalize_mask_uint8(mask_np)
    area = float(np.count_nonzero(mask_u8))

    # Reuse contours for polygon & perimeter
    contour, area_cv = _largest_contour(mask_u8)
    if contour is None:
        logger.debug("sahi_result_to_segmentation: no contours; fallback")
        centroid = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        perimeter = 0.0
        polygon = []
    else:
        peri = cv2.arcLength(contour, True)
        epsilon = 0.005 * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = approx.reshape(-1, 2).astype(np.float32).tolist()

        # Centroid from moments
        M = cv2.moments(mask_u8)
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        perimeter = float(peri)

    return Segmentation(
        id=0,
        box_id=0,
        mask=None,
        polygon=polygon,
        bbox=bbox,
        area_pixels=area,
        confidence=obj.score.value,
        centroid=centroid,
        perimeter_pixels=perimeter,
        source='yolo',
        class_id=getattr(getattr(obj, 'category', None), 'id', 0)
    )


# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------

def clip_polygon_to_bbox(polygon: List[List[float]], bbox: Tuple[float, float, float, float]) -> List[List[float]]:
    """
    Clip a polygon to a bounding box using Shapely intersection.
    bbox: (x, y, w, h)
    """
    try:
        minx, miny, width, height = bbox
        maxx = minx + width
        maxy = miny + height

        orig_poly = Polygon(polygon)
        if not orig_poly.is_valid:
            orig_poly = orig_poly.buffer(0)

        bbox_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        clipped = orig_poly.intersection(bbox_poly)

        if clipped.is_empty:
            return []
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if not isinstance(clipped, Polygon):
            return []

        coords = list(clipped.exterior.coords[:-1])
        if len(coords) < 3 or Polygon(coords).area < 1.0:
            return []

        return [[float(x), float(y)] for x, y in coords]

    except Exception as e:
        logger.error(f"clip_polygon_to_bbox: {e}")
        return []
