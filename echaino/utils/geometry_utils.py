# FILE: utils\geometry_utils.py
# PATH: D:\urchinScanner\utils\geometry_utils.py

import numpy as np
import cv2
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, MultiPolygon  # NEW: Import Shapely
from shapely import union_all  # NEW: Correct import for union_all (Shapely 2.0+)
import logging

from data.models import Segmentation

logger = logging.getLogger(__name__)


def mask_to_polygon(mask: Optional[np.ndarray], simplify: bool = True) -> List[List[float]]:
    """Convert binary mask to polygon coordinates"""
    if mask is None:
        logger.warning("None mask provided to mask_to_polygon")
        return []

    try:
        # Threshold if float
        if np.issubdtype(mask.dtype, np.floating):
            mask = (mask > 0.5).astype(np.uint8)
        elif mask.dtype == np.bool_:
            mask = mask.astype(np.uint8)
        elif mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Check if mask is empty
        if not np.any(mask):
            logger.warning("Empty mask provided to mask_to_polygon")
            return []

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning("No contours found in mask")
            return []

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Check minimum area
        area = cv2.contourArea(contour)
        if area < 10:  # Minimum area threshold
            logger.warning(f"Contour area too small: {area}")
            return []

        # Simplify if requested
        if simplify:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to list of points
        polygon = contour.reshape(-1, 2).tolist()

        # Ensure polygon has at least 3 points
        if len(polygon) < 3:
            logger.warning(f"Polygon has only {len(polygon)} points")
            return []

        return polygon

    except Exception as e:
        logger.error(f"Error converting mask to polygon: {e}")
        import traceback
        traceback.print_exc()
        return []


def calculate_bbox(mask: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
    """Calculate bounding box from mask"""
    if mask is None:
        return (0.0, 0.0, 0.0, 0.0)

    # Threshold if float
    if np.issubdtype(mask.dtype, np.floating):
        mask = (mask > 0.5)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return (0.0, 0.0, 0.0, 0.0)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return (float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1))


def calculate_area(mask: Optional[np.ndarray]) -> float:
    """Calculate area of mask in pixels"""
    if mask is None:
        return 0.0

    # Threshold if float
    if np.issubdtype(mask.dtype, np.floating):
        return float(np.sum(mask > 0.5))
    else:
        return float(np.sum(mask))


def polygon_to_mask(polygon: List[List[float]], shape: Tuple[int, int]) -> np.ndarray:
    """Convert polygon to binary mask"""
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def simplify_polygon(polygon: List[List[float]], tolerance: float = 1.0) -> List[List[float]]:
    """Simplify polygon using Douglas-Peucker algorithm"""
    try:
        poly = Polygon(polygon)
        simplified = poly.simplify(tolerance, preserve_topology=True)
        return list(simplified.exterior.coords[:-1])  # Remove duplicate last point
    except Exception as e:
        logger.error(f"Error simplifying polygon: {e}")
        return polygon


def calculate_polygon_iou(polygon1: List[List[float]], polygon2: List[List[float]]) -> float:
    """Calculate IoU between two polygons using Shapely"""
    try:
        poly1 = Polygon(polygon1)
        poly2 = Polygon(polygon2)

        # Fix invalid polygons
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        logger.warning(f"Polygon IoU calculation failed: {e}")
        return 0.0


def union_polygons(polygons: List[List[List[float]]]) -> List[List[float]]:
    """Union multiple polygons using Shapely, handling MultiPolygon and artifacts"""
    try:
        if not polygons:
            return []

        shapely_polys = []
        for poly in polygons:
            if not poly or len(poly) < 3:
                continue  # Skip empty or invalid polygons
            p = Polygon(poly)
            if not p.is_valid:
                p = p.buffer(0)  # Fix invalid geometry
            if p.is_valid and p.area > 0:
                shapely_polys.append(p)

        if not shapely_polys:
            return []

        # Union all
        unioned = union_all(shapely_polys)

        # Handle MultiPolygon: Take largest by area
        if isinstance(unioned, MultiPolygon):
            largest = max(unioned.geoms, key=lambda g: g.area)
            unioned = largest

        # Simplify to remove tiny artifacts
        unioned = unioned.simplify(1.0, preserve_topology=True)

        # Validate final
        if not unioned.is_valid or unioned.area < 10:
            logger.warning("Union resulted in invalid or tiny polygon")
            return []

        return list(unioned.exterior.coords[:-1])  # List of [x,y] points

    except Exception as e:
        logger.error(f"Polygon union failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# NEW: YOLO format conversion
def polygon_to_yolo_format(polygon: List[List[float]], img_w: int, img_h: int, class_id: int = 0) -> str:
    """Convert polygon to YOLO seg label string"""
    normalized = [f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in polygon]
    return f"{class_id} {' '.join(normalized)}"  # NEW: Prefix with class_id (default 0)


# NEW: SAHI result to Segmentation
def sahi_result_to_segmentation(obj) -> Segmentation:
    import cv2  # Ensure import if not already
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)

    bbox = obj.bbox.to_xywh()  # (x, y, w, h) as tuple
    mask_np = obj.mask.bool_mask if obj.mask else None
    if mask_np is None:
        logger.debug("No mask; using bbox approximations")
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
            class_id=obj.category.id if hasattr(obj, 'category') and hasattr(obj.category, 'id') else 0  # NEW: Set class_id, default 0
        )

    # Convert to uint8 once for all operations
    mask_uint8 = mask_np.astype(np.uint8)

    # Compute area (cheap NumPy op)
    area = float(np.sum(mask_np))

    # Compute contours ONCE and reuse
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No contours found; falling back to approximations")
        centroid = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        perimeter = 0.0
        polygon = []
    else:
        largest_contour = max(contours, key=cv2.contourArea)  # Use largest for robustness
        perimeter = cv2.arcLength(largest_contour, True)

        # Centroid from moments on mask (reuses mask_uint8)
        M = cv2.moments(mask_uint8)
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] != 0 else (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

        # Polygon from contours (simplify with approxPolyDP for speed; adjust epsilon for detail)
        epsilon = 0.005 * perimeter  # Lower for faster, less accurate polygons
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        polygon = approx_contour.reshape(-1, 2).tolist()

    # Discard temps explicitly for memory
    del mask_uint8, contours

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
        class_id=obj.category.id if hasattr(obj, 'category') and hasattr(obj.category, 'id') else 0  # NEW: Set class_id, default 0
    )


# NEW: Clip polygon to bounding box using Shapely
def clip_polygon_to_bbox(polygon: List[List[float]], bbox: Tuple[float, float, float, float]) -> List[List[float]]:
    """Clip a polygon to a bounding box using Shapely intersection."""
    try:
        minx, miny, width, height = bbox
        maxx = minx + width
        maxy = miny + height

        # Create Shapely Polygon for original and bbox
        orig_poly = Polygon(polygon)
        if not orig_poly.is_valid:
            orig_poly = orig_poly.buffer(0)  # Fix invalid

        bbox_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

        # Intersect
        clipped = orig_poly.intersection(bbox_poly)

        # Handle results
        if clipped.is_empty:
            return []
        elif isinstance(clipped, MultiPolygon):
            # Take largest
            largest = max(clipped.geoms, key=lambda g: g.area)
            clipped = largest
        elif not isinstance(clipped, Polygon):
            return []

        clipped_coords = list(clipped.exterior.coords[:-1])  # Remove closing point

        # Validate: min 3 points, positive area
        if len(clipped_coords) < 3 or Polygon(clipped_coords).area < 1.0:
            return []

        return clipped_coords

    except Exception as e:
        logger.error(f"Error clipping polygon: {e}")
        return []