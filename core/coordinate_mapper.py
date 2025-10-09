import numpy as np
from typing import Tuple, List, Optional
from data.models import Tile
import cv2

import logging
logger = logging.getLogger(__name__)


class CoordinateMapper:
    """Handles all coordinate transformations including SAM transforms"""

    def __init__(self, image_shape: Tuple[int, int]):
        """
        Args:
            image_shape: (height, width) of the full image
        """
        self.image_height, self.image_width = image_shape
        self.sam_size = 1024  # SAM's internal processing size

    def point_to_tile(self, point: Tuple[float, float],
                      tile: Tile) -> Tuple[float, float]:
        """Convert image point to tile coordinates"""
        x, y = point
        tile_x = x - tile.x
        tile_y = y - tile.y
        return (tile_x, tile_y)

    def tile_to_image(self, tile_point: Tuple[float, float],
                      tile: Tile) -> Tuple[float, float]:
        """Convert tile coordinates to image coordinates"""
        tx, ty = tile_point
        x = tx + tile.x
        y = ty + tile.y
        return (x, y)

    def mask_tile_to_image(self, tile_mask: np.ndarray,
                           tile: Tile) -> np.ndarray:
        full_mask = np.zeros((self.image_height, self.image_width), dtype=tile_mask.dtype)
        y1 = tile.y
        y2 = min(tile.y + tile.height, self.image_height)
        x1 = tile.x
        x2 = min(tile.x + tile.width, self.image_width)
        ty1 = 0
        ty2 = min(tile_mask.shape[0], y2 - y1)
        tx1 = 0
        tx2 = min(tile_mask.shape[1], x2 - x1)
        if (y2 - y1) != (ty2 - ty1) or (x2 - x1) != (tx2 - tx1):
            logger.debug("Dimension mismatch; resizing mask")
            target_h = y2 - y1
            target_w = x2 - x1
            tile_mask_resized = cv2.resize(
                tile_mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )
            full_mask[y1:y2, x1:x2] = tile_mask_resized > 0  # IMPROVED: Threshold for bool
        else:
            full_mask[y1:y2, x1:x2] = tile_mask[ty1:ty2, tx1:tx2]
        return full_mask

    def polygon_tile_to_image(self, tile_polygon: List[List[float]],
                              tile: Tile) -> List[List[float]]:
        """Convert polygon from tile to image coordinates"""
        image_polygon = []
        for x, y in tile_polygon:
            img_x = x + tile.x
            img_y = y + tile.y
            image_polygon.append([img_x, img_y])
        return image_polygon

    def calculate_sam_transform_params(self, original_shape: Tuple[int, int]) -> dict:
        """Calculate parameters for SAM's coordinate transformation"""
        old_h, old_w = original_shape

        # SAM resizes to have longest side = sam_size
        scale = self.sam_size * 1.0 / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale

        return {
            'scale': scale,
            'new_h': int(new_h + 0.5),
            'new_w': int(new_w + 0.5),
            'old_h': old_h,
            'old_w': old_w
        }

    def sam_coords_to_original(self, sam_coords: np.ndarray,
                               original_shape: Tuple[int, int]) -> np.ndarray:
        """Transform coordinates from SAM space back to original image space"""
        params = self.calculate_sam_transform_params(original_shape)
        coords = sam_coords.copy()
        coords = coords / params['scale']
        return coords

    def original_coords_to_sam(self, coords: np.ndarray,
                               original_shape: Tuple[int, int]) -> np.ndarray:
        """Transform coordinates from original image space to SAM space"""
        params = self.calculate_sam_transform_params(original_shape)
        coords = coords.copy()
        coords = coords * params['scale']
        return coords

    def point_in_sam_mask(self, point: Tuple[float, float],
                          sam_mask: np.ndarray,
                          original_shape: Tuple[int, int]) -> bool:
        """
        Check if a point (in original coordinates) is inside a SAM mask.

        Args:
            point: (x, y) in original image coordinates
            sam_mask: Mask from SAM output (typically 256x256)
            original_shape: (height, width) of the original image

        Returns:
            bool: Whether the point is inside the mask
        """
        # First resize the SAM mask to original dimensions
        mask_resized = cv2.resize(
            sam_mask.astype(np.uint8),
            (original_shape[1], original_shape[0]),  # width, height
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Check if point is in mask
        px, py = int(round(point[0])), int(round(point[1]))

        if 0 <= py < mask_resized.shape[0] and 0 <= px < mask_resized.shape[1]:
            return mask_resized[py, px]
        return False

    def resize_sam_mask_to_original(self, sam_mask: np.ndarray,
                                    original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize SAM output mask to original image dimensions.

        Args:
            sam_mask: Mask from SAM (typically 256x256)
            original_shape: (height, width) of original image

        Returns:
            Resized mask matching original dimensions
        """
        return cv2.resize(
            sam_mask.astype(np.uint8),
            (original_shape[1], original_shape[0]),  # width, height
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    def validate_point_in_tile(self, point: Tuple[float, float],
                               tile: Tile) -> bool:
        """Check if a point is within tile bounds"""
        x, y = point
        return (tile.x <= x < tile.x + tile.width and
                tile.y <= y < tile.y + tile.height)

    def image_to_display(self, point: Tuple[float, float],
                         scale: float,
                         offset: Tuple[float, float]) -> Tuple[float, float]:
        """Convert image coordinates to display coordinates"""
        x, y = point
        ox, oy = offset
        display_x = x * scale + ox
        display_y = y * scale + oy
        return (display_x, display_y)

    def display_to_image(self, display_point: Tuple[float, float],
                         scale: float,
                         offset: Tuple[float, float]) -> Tuple[float, float]:
        """Convert display coordinates to image coordinates"""
        dx, dy = display_point
        ox, oy = offset
        image_x = (dx - ox) / scale
        image_y = (dy - oy) / scale
        return (image_x, image_y)