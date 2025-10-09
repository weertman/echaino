# FILE: core\tile_manager.py
# PATH: D:\urchinScanner\core\tile_manager.py

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from data.models import BoundingBox, Tile, Segmentation
import math
import cv2
import os
import logging

logger = logging.getLogger(__name__)


class AdaptiveTileManager:
    """Generates optimal tiles around box clusters"""

    def __init__(self, config: dict):
        self.base_tile_size = config['tiling']['base_tile_size']
        self.overlap_ratio = config['tiling']['overlap_ratio']
        self.debug_mode = config.get('debug', False)

    def generate_tiles(self, image_shape: Tuple[int, int],
                       boxes: List[BoundingBox]) -> List[Tile]:
        """Generate a simple overlapping grid of tiles covering all boxes"""
        if not boxes:
            return []

        h, w = image_shape
        tile_size = self.base_tile_size
        overlap = int(tile_size * self.overlap_ratio)
        step = max(1, tile_size - overlap)

        # Find bounding box of all boxes with padding
        x_coords = []
        y_coords = []
        for box in boxes:
            x_coords.extend([box.x1, box.x2])
            y_coords.extend([box.y1, box.y2])

        min_x = max(0, int(min(x_coords)) - tile_size // 2)
        max_x = min(w, int(max(x_coords)) + tile_size // 2)
        min_y = max(0, int(min(y_coords)) - tile_size // 2)
        max_y = min(h, int(max(y_coords)) + tile_size // 2)

        tiles = []
        tile_id = 0

        # Generate grid covering the bounding box
        for y in range(min_y, max_y, step):
            for x in range(min_x, max_x, step):
                # Calculate tile bounds
                tile_x = x
                tile_y = y
                tile_w = min(tile_size, w - x)
                tile_h = min(tile_size, h - y)

                # Create tile bounding box for intersection testing
                tile_box = BoundingBox(-1, tile_x, tile_y,
                                       tile_x + tile_w, tile_y + tile_h)

                # Find boxes that intersect this tile
                tile_boxes = []
                for box in boxes:
                    if tile_box.intersects(box):
                        tile_boxes.append(box)

                # Only create tile if it has boxes
                if tile_boxes:
                    tile = Tile(
                        id=tile_id,
                        x=tile_x,
                        y=tile_y,
                        width=tile_w,
                        height=tile_h,
                        boxes=tile_boxes
                    )
                    tiles.append(tile)
                    tile_id += 1

        logger.info(f"Generated {len(tiles)} tiles for {len(boxes)} boxes")
        logger.info(f"  Tile size: {tile_size}, Overlap: {overlap}, Step: {step}")

        # Debug visualization
        if self.debug_mode:
            self._debug_save_tiles(image_shape, tiles, boxes)

        return tiles

    def generate_fixed_tiles(self, image_shape: Tuple[int, int],
                             annotations: List[Union[BoundingBox, Segmentation]],
                             tile_size: int = 640) -> List[Tile]:
        """Generate fixed 640x640 tiles focused on annotated regions"""
        h, w = image_shape
        tiles = []
        tile_id = 0

        # Find overall bounds with padding
        if annotations:
            x_coords = [a.bbox[0] for a in annotations] + [a.bbox[0] + a.bbox[2] for a in annotations]
            y_coords = [a.bbox[1] for a in annotations] + [a.bbox[1] + a.bbox[3] for a in annotations]

            min_x = max(0, math.floor(min(x_coords) - tile_size / 2))
            max_x = min(w, math.ceil(max(x_coords) + tile_size / 2))
            min_y = max(0, math.floor(min(y_coords) - tile_size / 2))
            max_y = min(h, math.ceil(max(y_coords) + tile_size / 2))
        else:
            min_x, min_y, max_x, max_y = 0, 0, w, h

        step = tile_size  # Non-overlapping for training simplicity

        for y in range(min_y, max_y, step):
            for x in range(min_x, max_x, step):
                tile_w = min(tile_size, w - x)
                tile_h = min(tile_size, h - y)
                tile_box = BoundingBox(-1, x, y, x + tile_w, y + tile_h)

                # Filter annotations intersecting this tile
                tile_anns = [
                    ann for ann in annotations
                    if tile_box.intersects(
                        ann if isinstance(ann, BoundingBox)
                        else BoundingBox(-1, ann.bbox[0], ann.bbox[1], ann.bbox[0] + ann.bbox[2],
                                         ann.bbox[1] + ann.bbox[3])
                    )
                ]

                if tile_anns:  # Only include tiles with annotations
                    tile = Tile(id=tile_id, x=x, y=y, width=tile_w, height=tile_h)
                    if annotations and isinstance(annotations[0], Segmentation):
                        tile.segmentations = tile_anns
                    else:
                        tile.boxes = tile_anns
                    tiles.append(tile)
                    tile_id += 1

        logger.info(f"Generated {len(tiles)} fixed tiles for training")
        return tiles

    def _debug_save_tiles(self, image_shape: Tuple[int, int],
                          tiles: List[Tile], boxes: List[BoundingBox]):
        """Save tile visualization for debugging"""
        # Create blank image
        h, w = image_shape
        scale = 1.0
        if max(h, w) > 4096:
            scale = 4096 / max(h, w)
            h = int(h * scale)
            w = int(w * scale)

        vis_image = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Draw tiles
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        for i, tile in enumerate(tiles):
            color = colors[i % len(colors)]
            x1 = int(tile.x * scale)
            y1 = int(tile.y * scale)
            x2 = int((tile.x + tile.width) * scale)
            y2 = int((tile.y + tile.height) * scale)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw boxes
        for box in boxes:
            x1 = int(box.x1 * scale)
            y1 = int(box.y1 * scale)
            x2 = int(box.x2 * scale)
            y2 = int(box.y2 * scale)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 0), 1)

        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "tile_visualization.png"), vis_image)
        logger.info("Saved tile visualization to debug_output/tile_visualization.png")