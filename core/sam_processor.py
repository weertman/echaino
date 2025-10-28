import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from data.models import Tile, BoundingBox, Segmentation
from utils.geometry_utils import mask_to_polygon, calculate_bbox, calculate_area
from core.coordinate_mapper import CoordinateMapper
import cv2
import logging
import os
import threading  # NEW: For lock
import gc  # NEW: For garbage collection

logger = logging.getLogger(__name__)


class SAMProcessor:
    """Handles SAM inference with box prompts - surgical approach"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['sam']['device'])
        self.model_type = config['sam']['model_type']
        self.checkpoint_path = config['sam']['checkpoint']
        self.predictor = None  # NEW: Start unloaded (deferred loading)
        self._model_loaded = False  # NEW: Flag to track state
        self._load_lock = threading.Lock()  # NEW: For thread safety

        # Processing settings
        self.batch_size = config['processing']['batch_size']
        self.debug_mode = config.get('debug', False)
        self.use_multimask = config.get('sam', {}).get('use_multimask', False)
        self.min_mask_area = config.get('sam', {}).get('min_mask_area', 1)  # Lowered default
        self.strict_clip = config.get('sam', {}).get('strict_clip', True)  # NEW: Option to force clipping

        # Create a transform object for coordinate handling
        # Note: This is safe as it doesn't depend on the model
        self.transform = ResizeLongestSide(1024)  # Assuming default SAM size; adjust if needed

    def load_model(self):
        """Load SAM model on-demand with thread safety"""
        with self._load_lock:
            if self._model_loaded:
                logger.debug("SAM model already loaded; skipping")
                return
            try:
                # Inline original _init_sam logic
                sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                self._model_loaded = True
                logger.info(f"Initialized SAM model: {self.model_type} on {self.device}")
                # Optional: Log memory usage
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    logger.debug(f"GPU memory after load: {mem:.2f} MB")
            except Exception as e:
                logger.error(f"Failed to initialize SAM: {e}")
                raise

    def unload_model(self):
        """Unload SAM model from GPU and free memory with best practices"""
        with self._load_lock:
            if not self._model_loaded:
                logger.debug("SAM model not loaded; skipping unload")
                return
            try:
                if self.predictor is not None:
                    # Move to CPU first (best practice)
                    self.predictor.model.cpu()
                    # Dereference
                    del self.predictor
                    self.predictor = None
                # Clear any lingering tensors
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()  # Call twice as recommended in some cases
                    torch.cuda.empty_cache()
                    mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    logger.debug(f"GPU memory after unload: {mem:.2f} MB")
                self._model_loaded = False
                logger.info("Unloaded SAM model from GPU")
            except Exception as e:
                logger.warning(f"Error during unload: {e}; memory may not be fully freed")

    def process_tile(self, tile_image: np.ndarray,
                     boxes: List[BoundingBox],
                     box_ids: List[int],
                     tile: Optional[Tile] = None) -> List[Dict]:
        """Process a single tile with SAM using box prompts"""
        if not self._model_loaded:
            self.load_model()  # NEW: Auto-load if needed
        if not boxes:
            return []

        # Store original tile dimensions
        orig_h, orig_w = tile_image.shape[:2]

        # Set image - SAM will handle all resizing internally
        self.predictor.set_image(tile_image)

        results = []

        for idx, (box, box_id) in enumerate(zip(boxes, box_ids)):
            try:
                # Ensure box coords are ordered
                x1, y1, x2, y2 = min(box.x1, box.x2), min(box.y1, box.y2), max(box.x1, box.x2), max(box.y1, box.y2)
                box = BoundingBox(box.id, x1, y1, x2, y2)

                # Add padding to box (10% on each side)
                pad_ratio = 0.1
                width = box.x2 - box.x1
                height = box.y2 - box.y1
                pad_x = width * pad_ratio
                pad_y = height * pad_ratio

                padded_box = BoundingBox(
                    box.id,
                    max(0, box.x1 - pad_x),
                    max(0, box.y1 - pad_y),
                    min(orig_w, box.x2 + pad_x),
                    min(orig_h, box.y2 + pad_y)
                )

                # Convert padded box to numpy array
                box_np = np.array([[padded_box.x1, padded_box.y1, padded_box.x2, padded_box.y2]], dtype=np.float32)

                # Initial predict with box prompt
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_np[0],
                    multimask_output=self.use_multimask
                )

                if len(masks) == 0:
                    logger.warning(f"No masks from box prompt for box {box_id}; trying center point fallback")
                    center_x = (box.x1 + box.x2) / 2
                    center_y = (box.y1 + box.y2) / 2
                    point_coords = np.array([[center_x, center_y]])
                    point_labels = np.array([1])  # Positive point
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_np[0],
                        multimask_output=self.use_multimask
                    )
                    if len(masks) == 0:
                        logger.error(f"SAM failed even with point fallback for box {box_id}")
                        results.append({
                            'box_id': box_id,
                            'mask': None,
                            'score': 0.0,
                            'box': box,
                            'tile': tile,
                            'raw_masks': [],
                            'selected_idx': None
                        })
                        continue

                # Select best mask
                if self.use_multimask:
                    best_idx = np.argmax(scores)
                    best_mask = masks[best_idx]
                    best_score = scores[best_idx]
                else:
                    best_mask = masks[0]
                    best_score = scores[0]

                # Clip mask to original (non-padded) box bounds
                clipped_mask = np.zeros_like(best_mask, dtype=bool)
                bx1 = int(max(0, box.x1))
                by1 = int(max(0, box.y1))
                bx2 = int(min(orig_w, box.x2))
                by2 = int(min(orig_h, box.y2))
                clipped_mask[by1:by2, bx1:bx2] = best_mask[by1:by2, bx1:bx2]

                area = int(np.sum(clipped_mask))
                if area < self.min_mask_area and self.strict_clip:
                    logger.warning(
                        f"Clipped mask area too small ({area} < {self.min_mask_area}) for box {box_id}; using full unclipped mask")
                    clipped_mask = best_mask
                    area = int(np.sum(clipped_mask))

                if area < self.min_mask_area:
                    logger.warning(f"Mask area still too small ({area}) for box {box_id}; discarding")
                    results.append({
                        'box_id': box_id,
                        'mask': None,
                        'score': float(best_score),
                        'box': box,
                        'tile': tile,
                        'raw_masks': [{'mask': best_mask, 'score': float(best_score), 'area': area}],
                        'selected_idx': 0
                    })
                    continue

                raw_masks = [{'mask': m, 'score': float(s), 'area': int(np.sum(m))} for m, s in zip(masks, scores)]

                results.append({
                    'box_id': box_id,
                    'mask': clipped_mask,
                    'score': float(best_score),
                    'box': box,
                    'tile': tile,
                    'raw_masks': raw_masks,
                    'selected_idx': 0 if not self.use_multimask else int(best_idx)
                })

            except Exception as e:
                logger.error(f"Error processing box {box_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'box_id': box_id,
                    'mask': None,
                    'score': 0.0,
                    'box': box,
                    'tile': tile,
                    'raw_masks': [],
                    'selected_idx': None
                })

        return results

    def batch_process_tiles(self, image: np.ndarray,
                            tiles: List[Tile],
                            coordinate_mapper: CoordinateMapper,
                            progress_callback: Optional[Callable] = None) -> List[Segmentation]:
        """Process multiple tiles and merge results"""
        if not self._model_loaded:
            self.load_model()  # NEW: Auto-load if needed for batch
        all_segmentations = []
        total_tiles = len(tiles)

        for i, tile in enumerate(tiles):
            if progress_callback:
                progress_callback(i, total_tiles, f"Processing tile {i + 1}/{total_tiles} with SAM")

            # Extract tile from image
            y1, y2 = tile.y, min(tile.y + tile.height, image.shape[0])
            x1, x2 = tile.x, min(tile.x + tile.width, image.shape[1])
            tile_image = image[y1:y2, x1:x2].copy()

            # Convert boxes to tile coordinates
            tile_boxes = []
            box_ids = []
            original_boxes = {}

            for box in tile.boxes:
                original_boxes[box.id] = box
                clipped_box = tile.clip_box(box)
                if clipped_box:
                    tile_box = BoundingBox(
                        box.id,
                        clipped_box.x1 - tile.x,
                        clipped_box.y1 - tile.y,
                        clipped_box.x2 - tile.x,
                        clipped_box.y2 - tile.y
                    )
                    tile_boxes.append(tile_box)
                    box_ids.append(box.id)

            # Process tile
            try:
                tile_results = self.process_tile(tile_image, tile_boxes, box_ids, tile)

                for result in tile_results:
                    if result['mask'] is None:
                        logger.debug(f"Skipping box {result['box_id']} - no mask")
                        continue

                    original_box = original_boxes.get(result['box_id'])
                    if not original_box:
                        logger.warning(f"Could not find original box for box_id {result['box_id']}")
                        continue

                    full_mask = coordinate_mapper.mask_tile_to_image(result['mask'], tile)

                    # Final global clip if strict_clip (but since process_tile already handled, optional)
                    if self.strict_clip:
                        box_mask = np.zeros_like(full_mask)
                        box_y1 = int(max(0, original_box.y1))
                        box_y2 = int(min(image.shape[0], original_box.y2))
                        box_x1 = int(max(0, original_box.x1))
                        box_x2 = int(min(image.shape[1], original_box.x2))
                        box_mask[box_y1:box_y2, box_x1:box_x2] = True
                        full_mask = full_mask & box_mask

                    mask_area = np.sum(full_mask)
                    if mask_area == 0:
                        logger.warning(f"Final mask empty for box {result['box_id']}")
                        continue

                    polygon = mask_to_polygon(full_mask)
                    if not polygon:
                        logger.warning(f"Could not extract polygon for box {result['box_id']}")
                        continue

                    bbox = calculate_bbox(full_mask)
                    area = calculate_area(full_mask)

                    M = cv2.moments(full_mask.astype(np.uint8))
                    if M["m00"] > 0:
                        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                    else:
                        centroid = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

                    segmentation = Segmentation(
                        id=len(all_segmentations),
                        box_id=result['box_id'],
                        mask=full_mask,
                        polygon=polygon,
                        bbox=bbox,
                        area_pixels=area,
                        confidence=result['score'],
                        centroid=centroid
                    )

                    all_segmentations.append(segmentation)

                    logger.debug(f"Created segmentation for box {result['box_id']}: area={area}, confidence={result['score']:.3f}")

            except Exception as e:
                logger.error(f"Error processing tile {tile.id}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if self.device.type == 'cuda' and (i + 1) % self.batch_size == 0:
                torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(total_tiles, total_tiles, "Processing tiles with SAM complete")

        logger.info(f"batch_process_tiles completed: {len(all_segmentations)} SAM segmentations created")

        return all_segmentations
