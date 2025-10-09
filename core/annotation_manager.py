# FILE: core\annotation_manager.py
# PATH: D:\urchinScanner\core\annotation_manager.py

import numpy as np
from typing import List, Optional, Callable, Tuple
from data.models import BoundingBox, Segmentation, ProjectData, Tile
from core.tile_manager import AdaptiveTileManager
from core.sam_processor import SAMProcessor
from core.coordinate_mapper import CoordinateMapper
from core.measurement_engine import MeasurementEngine
from datetime import datetime
import logging
import cv2
import os
from typing import Dict
import gc  # Added for explicit memory cleanup

from utils.geometry_utils import mask_to_polygon, calculate_bbox, calculate_area, calculate_polygon_iou, \
    union_polygons  # Updated import: added union_polygons

from collections import defaultdict  # Added for grouping

import colorsys  # NEW: For generate_distinct_colors without cv2
from typing import Tuple  # NEW: For type hinting colors

logger = logging.getLogger(__name__)


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate visually distinct colors using colorsys (stdlib, no cv2/np dependency)"""
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


class Command:
    """Base class for undo/redo commands"""

    def execute(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


class AddBoxCommand(Command):
    def __init__(self, manager: 'AnnotationManager', box: BoundingBox):
        self.manager = manager
        self.box = box

    def execute(self):
        self.manager.boxes.append(self.box)
        self.manager.all_boxes.append(self.box)  # Added

    def undo(self):
        self.manager.boxes.remove(self.box)
        self.manager.all_boxes.remove(self.box)  # Added


class EditBoxCommand(Command):
    def __init__(self, manager: 'AnnotationManager', box_id: int, new_x1: float, new_y1: float, new_x2: float, new_y2: float, new_class_id: Optional[int] = None):
        self.manager = manager
        self.box_id = box_id
        self.old_coords = None
        self.new_coords = (new_x1, new_y1, new_x2, new_y2)
        self.old_class_id = None  # NEW
        self.new_class_id = new_class_id  # NEW

    def execute(self):
        box = next((b for b in self.manager.all_boxes if b.id == self.box_id), None)
        if box:
            self.old_coords = (box.x1, box.y1, box.x2, box.y2)
            self.old_class_id = box.class_id  # NEW
            box.x1, box.y1, box.x2, box.y2 = self.new_coords
            if self.new_class_id is not None:
                box.class_id = self.new_class_id
            self.manager._sync_annotations(box)  # Sync associated seg

    def undo(self):
        box = next((b for b in self.manager.all_boxes if b.id == self.box_id), None)
        if box:
            box.x1, box.y1, box.x2, box.y2 = self.old_coords
            if self.old_class_id is not None:
                box.class_id = self.old_class_id
            self.manager._sync_annotations(box)


class DeleteBoxCommand(Command):
    def __init__(self, manager: 'AnnotationManager', box_id: int):
        self.manager = manager
        self.box_id = box_id
        self.deleted_box = None
        self.deleted_seg = None

    def execute(self):
        box = next((b for b in self.manager.all_boxes if b.id == self.box_id), None)
        if box:
            self.deleted_box = box
            self.manager.all_boxes.remove(box)
            self.manager.boxes = [b for b in self.manager.boxes if b.id != self.box_id]  # If separate list
            self.deleted_seg = next((s for s in self.manager.segmentations if s.box_id == self.box_id), None)
            if self.deleted_seg:
                self.manager.segmentations.remove(self.deleted_seg)
            self.manager.is_dirty = True

    def undo(self):
        if self.deleted_box:
            self.manager.all_boxes.append(self.deleted_box)
            if self.deleted_box not in self.manager.boxes:
                self.manager.boxes.append(self.deleted_box)
            if self.deleted_seg:
                self.manager.segmentations.append(self.deleted_seg)
            self.manager.is_dirty = True


class AnnotationManager:
    """Central manager for annotation workflow with debugging support"""

    def __init__(self, config: dict):
        self.config = config
        self.boxes: List[BoundingBox] = []
        self.all_boxes: List[BoundingBox] = []  # Added
        self.segmentations: List[Segmentation] = []
        self.history: List[Command] = []
        self.history_index = -1
        self.is_dirty = False  # Added for auto-save

        # NEW: Classes for multi-class (from config)
        self.classes = config.get('classes', [])  # List[str], empty for single-class

        # NEW: Load or generate class colors
        self.class_colors: List[Tuple[int, int, int]] = [tuple(c) for c in config.get('ui', {}).get('class_colors', [])]  # FIXED: Convert lists to tuples
        if self.classes:
            num_classes = len(self.classes)
            if len(self.class_colors) < num_classes:
                auto_colors = generate_distinct_colors(num_classes - len(self.class_colors))
                self.class_colors.extend(auto_colors)
                logger.info(f"Auto-generated {len(auto_colors)} class colors")
            # Fallback: If still empty, use box_color for all
            if not self.class_colors:
                fallback_color = tuple(config.get('ui', {}).get('box_color', [255, 0, 0]))
                self.class_colors = [fallback_color] * num_classes
                logger.info(f"Using fallback color {fallback_color} for all {num_classes} classes")
        else:
            self.class_colors = []  # No classes, no colors
            logger.debug("No classes defined; skipping class colors")

        # Enable debug mode
        self.debug_mode = config.get('debug', False)
        if self.debug_mode:
            os.makedirs("debug_output", exist_ok=True)
            logger.info("Debug mode enabled - output will be saved to debug_output/")

        # Debug data storage
        self.debug_tile_images: Dict[int, np.ndarray] = {}
        self.debug_tile_results: Dict[int, List[Dict]] = {}
        self.last_processed_tiles: List[Tile] = []

        # Initialize components
        self.tile_manager = AdaptiveTileManager(config)
        self.sam_processor = SAMProcessor(config)
        self.measurement_engine = MeasurementEngine()

        self._next_box_id = 0
        self._next_segmentation_id = 0

        # IMPROVED: Option to disable dedup in surgical mode
        self.enable_deduplication = config.get('processing', {}).get('enable_deduplication', False)
        # NEW: Option to enable merging of partials
        self.enable_merge_partials = config.get('processing', {}).get('enable_merge_partials', True)

    def add_box(self, x1: float, y1: float, x2: float, y2: float, class_id: int = 0) -> BoundingBox:
        """Add a new box annotation"""
        if self.classes and class_id >= len(self.classes):  # NEW: Clamp to valid range
            logger.warning(f"Invalid class_id {class_id}; clamping to 0")
            class_id = 0
        box = BoundingBox(id=self._next_box_id, x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id)  # NEW: Set class_id
        self._next_box_id += 1

        # Create and execute command
        command = AddBoxCommand(self, box)
        self._execute_command(command)
        self.is_dirty = True  # Added

        logger.info(f"Added box {box.id} at ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f}) with class_id {class_id}")
        return box

    def _execute_command(self, command: Command):
        """Execute command and update history"""
        # Remove any commands after current position
        self.history = self.history[:self.history_index + 1]

        # Execute and add to history
        command.execute()
        self.history.append(command)
        self.history_index += 1

    def undo(self):
        """Undo last action"""
        if self.history_index >= 0:
            command = self.history[self.history_index]
            command.undo()
            self.history_index -= 1
            self.is_dirty = True  # Added

    def redo(self):
        """Redo action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            command = self.history[self.history_index]
            command.execute()
            self.is_dirty = True  # Added

    def clear_all(self):
        """Clear all annotations"""
        self.boxes.clear()
        self.all_boxes.clear()  # Added
        self.segmentations.clear()
        self.history.clear()
        self.history_index = -1
        self._next_box_id = 0
        self._next_segmentation_id = 0
        self.is_dirty = True  # Added

    def process_annotations(self, image: np.ndarray,
                            scale_px_per_cm: Optional[float] = None,
                            progress_callback: Optional[Callable] = None,
                            refine_yolo: bool = False) -> List[Segmentation]:  # UPDATED: New param
        try:
            self.sam_processor.load_model()  # NEW: Load on-demand before processing
            if refine_yolo:
                # Filter to only YOLO-origin boxes for refinement
                self.boxes = [b for b in self.all_boxes if b.source == 'yolo']
                logger.info("Refining YOLO detections with SAM")

            if not self.boxes:
                logger.warning("No boxes to process")
                return []

            logger.info(f"Processing {len(self.boxes)} boxes on image {image.shape}")
            logger.info(f"Configuration:")
            logger.info(f"  Tile size: {self.config['tiling']['base_tile_size']}")
            logger.info(f"  Debug mode: {self.debug_mode}")
            logger.info(f"  Using surgical approach: box → best mask")
            logger.info(f"  Merge partials: {self.enable_merge_partials}")

            # Clear previous debug data
            self.debug_tile_images.clear()
            self.debug_tile_results.clear()

            # Update measurement scale
            if scale_px_per_cm:
                self.measurement_engine.update_scale(scale_px_per_cm)

            # Create coordinate mapper
            coordinate_mapper = CoordinateMapper(image.shape[:2])

            # Step 1: Generate tiles
            if progress_callback:
                progress_callback(0, 100, "Generating tiles...")
            tiles = self.tile_manager.generate_tiles(image.shape[:2], self.boxes)
            logger.info(f"Generated {len(tiles)} tiles")

            # Store tiles for debug viewer
            self.last_processed_tiles = tiles

            # Extract and store tile images for debugging
            if self.debug_mode:
                logger.info("Storing debug tile images...")
                for tile in tiles:
                    try:
                        y1, y2 = tile.y, min(tile.y + tile.height, image.shape[0])
                        x1, x2 = tile.x, min(tile.x + tile.width, image.shape[1])
                        tile_image = image[y1:y2, x1:x2].copy()
                        self.debug_tile_images[tile.id] = tile_image
                        logger.debug(f"Stored debug image for tile {tile.id}: shape {tile_image.shape}")
                    except Exception as e:
                        logger.error(f"Failed to extract debug image for tile {tile.id}: {e}")

            # Step 2: Process tiles with SAM
            if progress_callback:
                progress_callback(20, 100, "Running segmentation...")

            # Process tiles and capture debug information
            segmentations = self._process_tiles_with_debug(
                image, tiles, coordinate_mapper,
                lambda i, t, m: progress_callback(20 + int(60 * i / t), 100, m) if progress_callback else None
            )

            logger.info(f"Generated {len(segmentations)} raw segmentations")

            # NEW: Merge partial segmentations by box_id if enabled
            if self.enable_merge_partials:
                if progress_callback:
                    progress_callback(70, 100, "Merging partial segmentations...")
                segmentations = self._merge_partial_segmentations(segmentations, image.shape[:2])
                logger.info(f"After merging: {len(segmentations)} segmentations")

            # Added: Mark statuses
            successful_ids = {seg.box_id for seg in segmentations}
            for box in self.all_boxes:
                if box.id in successful_ids:
                    box.status = "processed_success"
                else:
                    box.status = "processed_failed"

            # Step 3: Deduplicate masks (optional, now after merge)
            if self.enable_deduplication:
                if progress_callback:
                    progress_callback(80, 100, "Deduplicating masks...")
                segmentations = self._deduplicate_masks(segmentations)
                logger.info(f"After deduplication: {len(segmentations)} segmentations")

            # Assign IDs and store
            for i, seg in enumerate(segmentations):
                seg.id = self._next_segmentation_id + i
            self._next_segmentation_id += len(segmentations)

            self.segmentations = segmentations

            # Save debug visualization if enabled
            if self.debug_mode:
                self._save_debug_summary(image, tiles, segmentations)

            if progress_callback:
                progress_callback(100, 100, "Complete!")

            logger.info(f"Processing complete: {len(segmentations)} final segmentations")

            # Log summary of results (IMPROVED: More detailed if none generated)
            if segmentations:
                areas = [s.area_pixels for s in segmentations]
                scores = [s.confidence for s in segmentations]
                logger.info(f"Segmentation statistics:")
                logger.info(f"  Area range: {min(areas):.0f} - {max(areas):.0f} pixels")
                logger.info(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
            else:
                logger.warning("No segmentations were generated!")
                logger.warning("Possible causes:")
                logger.warning("  - SAM failed to find good masks for the boxes (check debug logs for discarded masks)")
                logger.warning("  - Boxes may not be properly aligned with objects")
                logger.warning("  - Try adjusting box placement or tile size")
                logger.warning("  - Check if strict_clip is False in config to allow spilled masks")

            # Added: Explicit memory cleanup
            gc.collect()

            self.is_dirty = True  # Added
            return segmentations
        finally:
            # NEW: Guaranteed unload, unless debug skip
            if not (self.debug_mode and self.config.get('sam', {}).get('keep_loaded_in_debug', False)):
                self.sam_processor.unload_model()
            else:
                logger.debug("Debug mode: Skipping SAM unload")

    # NEW: Method to add YOLO detections
    def add_yolo_detections(self, detections: List[Segmentation]):
        """Add detections from YOLO inference for refinement"""
        for det in detections:
            # Create box from segmentation bbox if needed
            box = BoundingBox(
                id=self._next_box_id,
                x1=det.bbox[0],
                y1=det.bbox[1],
                x2=det.bbox[0] + det.bbox[2],
                y2=det.bbox[1] + det.bbox[3],
                source='yolo',  # New: Set source
                class_id=det.class_id  # NEW: Propagate class_id from detection
            )
            self._next_box_id += 1
            command = AddBoxCommand(self, box)
            self._execute_command(command)

            # Add segmentation with YOLO flag and class info
            det.id = self._next_segmentation_id
            det.source = 'yolo'  # Set source
            # NEW: Set class_name if classes available
            if self.classes and det.class_id < len(self.classes):
                det.class_name = self.classes[det.class_id]
            self.segmentations.append(det)
            self._next_segmentation_id += 1

        self.is_dirty = True
        logger.info(f"Added {len(detections)} YOLO detections")

    def _merge_partial_segmentations(self, segmentations: List[Segmentation], image_shape: Tuple[int, int]) -> List[
        Segmentation]:
        """Merge partial segmentations sharing the same box_id using polygon union (fallback to mask)"""
        if len(segmentations) <= 1:
            return segmentations

        grouped = defaultdict(list)
        for seg in segmentations:
            # NEW: Only group if matching class_id (compare to first in group)
            group = grouped[seg.box_id]
            if group and seg.class_id != group[0].class_id:
                continue  # Skip if different class
            grouped[seg.box_id].append(seg)

        merged = []
        for box_id, group in grouped.items():
            if len(group) == 1:
                merged.append(group[0])
                # Update status to indicate single-tile success
                box = next((b for b in self.all_boxes if b.id == box_id), None)
                if box:
                    box.status = "processed_success"
                continue

            logger.info(f"Merging {len(group)} partials for box {box_id}")

            try:
                # Step 1: Polygon-based merge (preferred for low RAM)
                polys = [seg.polygon for seg in group if seg.polygon]
                if polys and all(len(p) >= 3 for p in polys):  # Ensure valid polygons
                    merged_polygon = union_polygons(polys)  # From utils/geometry_utils
                    if not merged_polygon:
                        raise ValueError("Polygon union failed")

                    # Calculate unioned bbox for temp mask (bounded to save RAM)
                    min_x = min(min(p[0] for p in poly) for poly in polys)
                    min_y = min(min(p[1] for p in poly) for poly in polys)
                    max_x = max(max(p[0] for p in poly) for poly in polys)
                    max_y = max(max(p[1] for p in poly) for poly in polys)
                    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

                    # Create temp small mask for measurements (relative to bbox)
                    mask_w = int(bbox[2]) + 1
                    mask_h = int(bbox[3]) + 1
                    temp_uint8 = np.zeros((mask_h, mask_w), dtype=np.uint8)  # Use uint8 directly for filling

                    # Translate polygon to mask-local coords and fill
                    pts = np.array([[px - min_x, py - min_y] for px, py in merged_polygon], dtype=np.int32)
                    cv2.fillPoly(temp_uint8, [pts], 1)  # Fill on uint8 array

                    temp_mask = temp_uint8.astype(bool)  # Convert to bool for measurements (or use >0 for safety)

                    # Derive properties from temp mask
                    area = calculate_area(temp_mask)
                    perimeter = self.measurement_engine._calculate_perimeter(temp_mask)
                    centroid_local = self.measurement_engine._calculate_centroid(temp_mask)
                    centroid = (centroid_local[0] + min_x, centroid_local[1] + min_y)  # Translate back

                    # Discard temp mask immediately
                    del temp_uint8  # Also del the uint8 version
                    del temp_mask
                    gc.collect()

                else:
                    # Fallback: Mask-based merge if no polygons
                    logger.warning(f"Falling back to mask merge for box {box_id} (no valid polygons)")
                    # Compute unioned bbox first to create small temp masks
                    bboxes = [seg.bbox for seg in group]  # FIXED: Use group
                    min_x = min(b[0] for b in bboxes)
                    min_y = min(b[1] for b in bboxes)
                    max_x = max(b[0] + b[2] for b in bboxes)
                    max_y = max(b[1] + b[3] for b in bboxes)  # FIXED: Correct max expression
                    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

                    merged_mask = np.zeros((int(bbox[3]) + 1, int(bbox[2]) + 1), dtype=bool)
                    for seg in group:
                        if seg.mask is None:
                            continue  # Skip if no mask (though should be rare)
                        # Clip and translate full mask to small merged_mask coords
                        sy1 = max(0, int(seg.bbox[1] - min_y))
                        sy2 = min(merged_mask.shape[0], sy1 + int(seg.bbox[3]))
                        sx1 = max(0, int(seg.bbox[0] - min_x))
                        sx2 = min(merged_mask.shape[1], sx1 + int(seg.bbox[2]))
                        merged_mask[sy1:sy2, sx1:sx2] = np.logical_or(
                            merged_mask[sy1:sy2, sx1:sx2],
                            seg.mask[int(seg.bbox[1]):int(seg.bbox[1] + seg.bbox[3]),
                            int(seg.bbox[0]):int(seg.bbox[0] + seg.bbox[2])]
                        )

                    # Clean and extract largest component
                    kernel = np.ones((3, 3), np.uint8)
                    merged_mask = cv2.morphologyEx(merged_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
                    labels, num = cv2.connectedComponents(merged_mask.astype(np.uint8))
                    if num > 1:
                        sizes = np.bincount(labels.ravel())[1:]
                        largest = np.argmax(sizes) + 1
                        merged_mask = (labels == largest)

                    merged_polygon = mask_to_polygon(merged_mask)
                    area = calculate_area(merged_mask)
                    perimeter = self.measurement_engine._calculate_perimeter(merged_mask)
                    centroid_local = self.measurement_engine._calculate_centroid(merged_mask)
                    centroid = (centroid_local[0] + min_x, centroid_local[1] + min_y)

                    # Discard temp mask
                    del merged_mask
                    gc.collect()

                # Create merged segmentation
                max_conf = max(seg.confidence for seg in group)
                merged_seg = Segmentation(
                    id=len(merged),  # Temporary ID, reassigned later
                    box_id=box_id,
                    mask=None,  # Do not keep full mask
                    polygon=merged_polygon,
                    bbox=bbox,
                    area_pixels=area,
                    confidence=max_conf,
                    centroid=centroid,
                    perimeter_pixels=perimeter,
                    class_id=group[0].class_id  # NEW: Set from group
                )

                # Compute additional measurements if needed
                merged_seg = self.measurement_engine.calculate_measurements(merged_seg)

                merged.append(merged_seg)

                # Update box status
                box = next((b for b in self.all_boxes if b.id == box_id), None)
                if box:
                    box.status = "merged_success"

            except Exception as e:
                logger.error(f"Merge failed for box {box_id}: {e}. Falling back to highest confidence.")
                # Fallback: Select highest confidence partial
                best_seg = max(group, key=lambda s: s.confidence)
                merged.append(best_seg)
                box = next((b for b in self.all_boxes if b.id == box_id), None)
                if box:
                    box.status = "processed_partial"

        return merged

    def _deduplicate_masks(self, segmentations: List[Segmentation]) -> List[Segmentation]:
        if len(segmentations) <= 1:
            return segmentations

        # Sort by confidence (keep higher confidence masks)
        segmentations.sort(key=lambda s: s.confidence, reverse=True)

        kept = []
        used_boxes = set()

        for seg in segmentations:
            # Skip if box already has a segmentation
            if seg.box_id in used_boxes:
                continue

            # Check overlap with existing masks
            overlap_found = False
            for kept_seg in kept:
                # NEW: Skip if different class_id
                if seg.class_id != kept_seg.class_id:
                    continue
                overlap = calculate_polygon_iou(seg.polygon, kept_seg.polygon)  # Updated to polygon IoU
                if overlap > 0.5:  # More than 50% overlap
                    overlap_found = True
                    break

            if not overlap_found:
                kept.append(seg)
                used_boxes.add(seg.box_id)

        # Added: Update statuses for deduped (removed) as failed
        for seg in segmentations:
            if seg not in kept:
                box = next((b for b in self.all_boxes if b.id == seg.box_id), None)
                if box:
                    box.status = "processed_failed"

        return kept

    def _save_debug_summary(self, image: np.ndarray, tiles: List[Tile],
                            segmentations: List[Segmentation]):
        """Save comprehensive debug summary"""
        debug_dir = "debug_output"

        # Create summary text with UTF-8 encoding
        with open(os.path.join(debug_dir, "processing_summary.txt"), 'w', encoding='utf-8') as f:
            f.write("Processing Summary - Surgical Approach\n")
            f.write("=====================================\n\n")
            f.write(f"Total boxes: {len(self.boxes)}\n")
            f.write(f"Total tiles: {len(tiles)}\n")
            f.write(f"Total segmentations: {len(segmentations)}\n\n")

            f.write("Configuration:\n")
            f.write(f"  Tile size: {self.config['tiling']['base_tile_size']}\n")
            f.write(f"  Approach: Best mask per box (highest confidence)\n\n")

            # Per-tile summary
            f.write("Per-Tile Results:\n")
            for tile_id, tile_results in self.debug_tile_results.items():
                tile = next((t for t in tiles if t.id == tile_id), None)
                if tile:
                    f.write(f"\nTile {tile_id} at ({tile.x}, {tile.y}):\n")
                    f.write(f"  Boxes in tile: {len(tile.boxes)}\n")

                    for result in tile_results:
                        f.write(f"  Box {result['box_id']}:\n")
                        if result.get('raw_masks'):
                            f.write(f"    SAM produced: {len(result['raw_masks'])} masks\n")
                            for i, rm in enumerate(result['raw_masks']):
                                selected = " [SELECTED]" if i == result.get('selected_idx') else ""
                                f.write(
                                    f"      Mask {i}: score={rm['score']:.3f}, area={rm['area']}{selected}\n")
                        else:
                            f.write(f"    No masks produced\n")

            # Failed boxes summary
            f.write("\n\nFailed Boxes:\n")
            successful_boxes = {seg.box_id for seg in segmentations}
            for box in self.boxes:
                if box.id not in successful_boxes:
                    f.write(f"  Box {box.id} at ({box.x1:.1f}, {box.y1:.1f}) to ({box.x2:.1f}, {box.y2:.1f})\n")

    def _process_tiles_with_debug(self, image: np.ndarray, tiles: List[Tile],
                                  coordinate_mapper: CoordinateMapper,
                                  progress_callback: Optional[Callable] = None) -> List[Segmentation]:
        """Process tiles and store debug information"""
        all_segmentations = []
        total_tiles = len(tiles)

        for i, tile in enumerate(tiles):
            if progress_callback:
                progress_callback(i, total_tiles, f"Processing tile {i + 1}/{total_tiles}")

            # Extract tile image
            y1, y2 = tile.y, min(tile.y + tile.height, image.shape[0])
            x1, x2 = tile.x, min(tile.x + tile.width, image.shape[1])
            tile_image = image[y1:y2, x1:x2].copy()

            # Convert boxes to tile coordinates
            tile_boxes = []
            box_ids = []
            for box in tile.boxes:
                # Clip box to tile boundaries
                clipped_box = tile.clip_box(box)
                if clipped_box:
                    # Convert to tile-relative coordinates
                    tile_box = BoundingBox(
                        box.id,
                        clipped_box.x1 - tile.x,
                        clipped_box.y1 - tile.y,
                        clipped_box.x2 - tile.x,  # FIXED: Remove duplicate - tile.x
                        clipped_box.y2 - tile.y
                    )
                    tile_boxes.append(tile_box)
                    box_ids.append(box.id)

            # Process tile
            try:
                # Process and get raw results
                tile_results = self.sam_processor.process_tile(
                    tile_image, tile_boxes, box_ids, tile
                )

                # Store debug results only if debug_mode
                if self.debug_mode:
                    self.debug_tile_results[tile.id] = tile_results

                # Log tile processing summary
                logger.info(f"Tile {tile.id} results:")
                for result in tile_results:
                    if result['mask'] is not None:
                        logger.info(f"  Box {result['box_id']}: Success (score={result['score']:.3f})")
                    else:
                        logger.info(f"  Box {result['box_id']}: Failed (no mask)")

                # Convert results to full segmentations
                for result in tile_results:
                    if result.get('mask') is None:
                        logger.debug(f"Skipping result for box {result['box_id']} - no mask")
                        # Added: Mark box as failed
                        box = next((b for b in self.all_boxes if b.id == result['box_id']), None)
                        if box:
                            box.status = "processed_failed"
                        continue

                    # Create full image mask
                    full_mask = coordinate_mapper.mask_tile_to_image(result['mask'], tile)

                    # Extract polygon
                    polygon = mask_to_polygon(full_mask)
                    if not polygon:
                        logger.warning(f"Could not extract polygon for box {result['box_id']}")
                        box = next((b for b in self.all_boxes if b.id == result['box_id']), None)
                        if box:
                            box.status = "processed_failed"
                        continue

                    # Calculate properties
                    bbox = calculate_bbox(full_mask)
                    area = calculate_area(full_mask)

                    # NEW: Get box for class_id
                    box = next((b for b in self.all_boxes if b.id == result['box_id']), None)
                    class_id = box.class_id if box else 0
                    class_name = self.classes[class_id] if self.classes and class_id < len(self.classes) else None

                    segmentation = Segmentation(
                        id=len(all_segmentations),
                        box_id=result['box_id'],
                        mask=full_mask,  # Temporary; will discard later if not keeping
                        polygon=polygon,
                        bbox=bbox,
                        area_pixels=area,
                        confidence=result['score'],
                        class_id=class_id,  # NEW
                        class_name=class_name  # NEW
                    )

                    # Compute measurements immediately (uses mask)
                    segmentation = self.measurement_engine.calculate_measurements(segmentation)

                    # Discard mask immediately unless configured to keep (but keep temp for merge later)
                    if not self.config.get('processing', {}).get('keep_full_masks', False):
                        segmentation.mask = None  # Discard now; merge will handle temp if needed

                    all_segmentations.append(segmentation)

            except Exception as e:
                logger.error(f"Error processing tile {tile.id}: {e}")
                import traceback
                traceback.print_exc()
                if self.debug_mode:
                    self.debug_tile_results[tile.id] = []
                continue

        return all_segmentations

    def get_project_data(self, image_path: str,
                         image_shape: Tuple[int, int, int],
                         scale_px_per_cm: Optional[float] = None) -> ProjectData:
        """Get current project data"""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'num_boxes': len(self.boxes),
            'num_segmentations': len(self.segmentations),
            'next_box_id': self._next_box_id,  # Added to persist ID counter
            'classes': self.classes  # NEW: Include classes list
        }
        return ProjectData(
            image_path=image_path,
            image_shape=image_shape,
            boxes=self.boxes.copy(),
            all_boxes=self.all_boxes.copy(),  # Added
            segmentations=self.segmentations.copy(),
            scale_px_per_cm=scale_px_per_cm,
            metadata=metadata
        )

    def restore_from_project(self, project: ProjectData):
        """Restore state from loaded project"""
        self.boxes = project.boxes.copy()
        self.all_boxes = project.all_boxes.copy()
        self.segmentations = project.segmentations.copy()
        self._next_box_id = project.metadata.get('next_box_id',
                                                 max(b.id for b in self.all_boxes) + 1 if self.all_boxes else 0)
        self._next_segmentation_id = max(s.id for s in self.segmentations) + 1 if self.segmentations else 0
        self.classes = project.metadata.get('classes', [])  # NEW: Restore classes
        # Use metadata flag if present
        is_legacy = project.metadata.get('is_legacy', False)  # NEW: Check for legacy flag from load_project
        if not self.classes and (self.all_boxes or self.segmentations):
            default_class = "urchin"  # Changed from "default" to "urchin" per request
            self.classes = [default_class]
            logger.info(f"Old project detected; setting single class '{default_class}'")
            is_legacy = True  # Set if not already

        # Assign class_id=0 and class_name if missing or legacy
        for box in self.all_boxes + self.boxes:
            if not hasattr(box, 'class_id') or box.class_id is None:
                box.class_id = 0
            if is_legacy:
                box.class_name = self.classes[0]  # NEW: Set class_name to "urchin"

        for seg in self.segmentations:
            if not hasattr(seg, 'class_id') or seg.class_id is None:
                seg.class_id = 0
            if is_legacy:
                seg.class_name = self.classes[0]  # NEW: Set class_name to "urchin"

        # NEW: Add a property or return value for UI; since this is manager, add attribute
        self.is_legacy_project = is_legacy  # NEW: Attribute for UI to check and trigger dialog

        # NEW: Recompute class_colors based on self.config
        self.class_colors = [tuple(c) for c in
                             self.config.get('ui', {}).get('class_colors', [])]  # FIXED: Convert to tuples
        if self.classes:
            num_classes = len(self.classes)
            if len(self.class_colors) < num_classes:
                auto_colors = generate_distinct_colors(num_classes - len(self.class_colors))
                self.class_colors.extend(auto_colors)
            if not self.class_colors:
                fallback_color = tuple(self.config.get('ui', {}).get('box_color', [255, 0, 0]))
                self.class_colors = [fallback_color] * num_classes
                logger.info(f"Restored with fallback color {fallback_color} for all {num_classes} classes")
        else:
            self.class_colors = []  # No classes, no colors
            logger.debug("No classes defined; skipping class colors")
        self.is_dirty = False

    def _sync_annotations(self, box: BoundingBox):
        seg = next((s for s in self.segmentations if s.box_id == box.id), None)
        if seg and seg.mask is not None:  # If mask exists, recalculate
            seg.bbox = calculate_bbox(seg.mask)
            seg.area_pixels = calculate_area(seg.mask)
            seg.polygon = mask_to_polygon(seg.mask)
        elif seg:  # If no mask, approximate from box
            seg.bbox = (box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
        if seg:  # NEW: Sync class_id if seg exists
            seg.class_id = box.class_id
        self.is_dirty = True

    def update_classes(self, new_classes):
        """Update class list at runtime, maintain colors, and clamp existing annotations."""
        try:
            if new_classes is None:
                return
            new_classes = list(new_classes)
            self.classes = new_classes

            # Persist into config (if caller saves it later)
            try:
                self.config['classes'] = list(new_classes)
            except Exception:
                pass

            # Rebuild/expand class_colors to match length
            current_colors = []
            try:
                current_colors = list(self.class_colors) if hasattr(self, 'class_colors') else []
            except Exception:
                current_colors = []
            # Normalize to tuples
            current_colors = [tuple(c) for c in current_colors if isinstance(c, (list, tuple)) and len(c) == 3]

            required = len(new_classes)
            if len(current_colors) < required:
                extra = generate_distinct_colors(required - len(current_colors))
                current_colors.extend(extra)
            elif len(current_colors) > required:
                current_colors = current_colors[:required]

            self.class_colors = current_colors
            # Mirror into config.ui.class_colors
            try:
                self.config.setdefault('ui', {})
                self.config['ui']['class_colors'] = [list(c) for c in current_colors]
            except Exception:
                pass

            # Clamp existing annotations
            if required == 0:
                for b in self.boxes:
                    b.class_id = 0
                for s in self.segmentations:
                    s.class_id = 0
            else:
                max_id = required - 1
                for b in self.boxes:
                    if getattr(b, 'class_id', 0) > max_id:
                        b.class_id = max_id
                for s in self.segmentations:
                    if getattr(s, 'class_id', 0) > max_id:
                        s.class_id = max_id

            self.is_dirty = True
            logger.info(f"AnnotationManager classes updated -> {new_classes} (colors: {len(self.class_colors)})")
        except Exception as e:
            logger.error(f"update_classes failed: {e}")
