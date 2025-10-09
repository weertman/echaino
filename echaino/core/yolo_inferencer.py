# FILE: core\yolo_inferencer.py
# PATH: D:\urchinScanner\core\yolo_inferencer.py

from ultralytics import YOLO
import numpy as np
import torch
from typing import List, Dict, Tuple
from data.models import BoundingBox, Segmentation
from utils.geometry_utils import sahi_result_to_segmentation
from sahi.predict import get_sliced_prediction
from sahi.slicing import slice_image
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi import AutoDetectionModel, ObjectPrediction
from sahi.postprocess.combine import NMSPostprocess
from tqdm import tqdm
import logging
import threading
import gc  # For memory cleanup
from ultralytics.utils import LOGGER as ultralytics_logger
import sys
import multiprocessing as mp  # Added for parallelism

# Suppress Ultralytics verbose output globally
ultralytics_logger.setLevel(logging.WARNING)  # Set to WARNING or ERROR to reduce verbosity

logger = logging.getLogger(__name__)

def process_pred(pred, roi_x1, roi_y1):
    det = sahi_result_to_segmentation(pred)
    det.bbox = (det.bbox[0] + roi_x1, det.bbox[1] + roi_y1, det.bbox[2], det.bbox[3])
    return det

class YoloInferencer:
    """Handles YOLO inference with SAHI for small object detection"""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.batch_size = max(1, config.get('yolo', {}).get('sahi_batch_size', 1))  # From config, min 1
        self.slice_height = config.get('sahi', {}).get('slice_size', 640)
        self.slice_width = config.get('sahi', {}).get('slice_size', 640)
        self.overlap_ratio = config.get('sahi', {}).get('overlap_ratio', 0.2)
        self.postprocess_threshold = config.get('sahi', {}).get('postprocess_match_threshold', 0.5)
        self.inference_lock = threading.Lock()  # For thread-safe GPU access
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, model_path: str):
        """Load YOLO model"""
        if self.model is None:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")

    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded YOLO model")

    def infer_on_roi(self, image: np.ndarray, roi: BoundingBox) -> List[Segmentation]:
        """Run SAHI inference on ROI with batch processing of slices"""
        x1, y1, x2, y2 = int(roi.x1), int(roi.y1), int(roi.x2), int(roi.y2)
        roi_image = image[y1:y2, x1:x2]
        roi_height, roi_width = roi_image.shape[:2]

        try:
            # Phase 1: Generate slices
            slice_result = slice_image(
                roi_image,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio
            )
            tile_images = slice_result.images  # List[np.ndarray]
            starting_pixels = slice_result.starting_pixels  # List[List[int]] [x,y]

            num_slices = len(tile_images)
            if num_slices < self.batch_size * 2:
                logger.info(f"Few slices ({num_slices}); falling back to sequential")
                return self._sequential_inference(roi_image, roi)

            all_predictions = []  # Collect ObjectPrediction

            # Phase 2-4: Batching and Prediction with tqdm for batches
            for i in tqdm(range(0, num_slices, self.batch_size), desc="Processing Batches"):
                batch_images = tile_images[i:i + self.batch_size]
                batch_starting = starting_pixels[i:i + self.batch_size]

                try:
                    # Pad batch to uniform size for efficient batching
                    max_h = max(img.shape[0] for img in batch_images)
                    max_w = max(img.shape[1] for img in batch_images)
                    padded_batch = []
                    pad_offsets = []  # (left, top, right, bottom) for each
                    for img in batch_images:
                        pad_bottom = max_h - img.shape[0]
                        pad_right = max_w - img.shape[1]
                        padded = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
                        padded_batch.append(padded)
                        pad_offsets.append((0, 0, pad_right, pad_bottom))

                    with self.inference_lock:
                        results = self.model(padded_batch, conf=0.25, iou=0.45, device=self.device, verbose=False)  # Batch predict

                    # Convert to SAHI ObjectPrediction, adjust for padding and slice offsets
                    for j, res in enumerate(results):
                        offset_x, offset_y = batch_starting[j]
                        pad_left, pad_top, pad_right, pad_bottom = pad_offsets[j]
                        tile_h, tile_w = batch_images[j].shape[:2]  # Original tile size
                        if res.boxes is not None:
                            for k in range(len(res.boxes)):
                                box = res.boxes.xyxy[k].cpu().numpy()
                                conf = res.boxes.conf[k].item()
                                cls_id = int(res.boxes.cls[k].item()) if res.boxes.cls is not None else 0  # NEW: Extract class_id, default 0
                                x1, y1, x2, y2 = box
                                # Depad coordinates
                                x1 = max(0, x1 - pad_left)
                                y1 = max(0, y1 - pad_top)
                                x2 = min(tile_w, x2 - pad_left)
                                y2 = min(tile_h, y2 - pad_top)
                                # Offset to slice position in ROI
                                x1 += offset_x
                                y1 += offset_y
                                x2 += offset_x
                                y2 += offset_y
                                coco_segmentation = None
                                if res.masks is not None:
                                    try:
                                        # Get mask, depad it
                                        mask_tensor = res.masks.data[k].cpu().numpy()  # (H_padded, W_padded)
                                        mask_tensor = mask_tensor[0:tile_h, 0:tile_w]  # Depad
                                        bool_mask = mask_tensor > 0.5
                                        if not np.any(bool_mask):  # Skip empty masks
                                            continue
                                        coco_segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                                        if not coco_segmentation or not coco_segmentation[0]:  # Skip invalid/empty
                                            continue
                                        # Offset the segmentation points
                                        offset_segmentation = []
                                        for seg in coco_segmentation:
                                            offset_seg = []
                                            for idx in range(0, len(seg), 2):
                                                offset_seg.append(seg[idx] + offset_x)
                                                offset_seg.append(seg[idx + 1] + offset_y)
                                            offset_segmentation.append(offset_seg)
                                        coco_segmentation = offset_segmentation
                                    except Exception as e:
                                        logger.warning(f"Skipping mask due to error: {e}")
                                        continue
                                try:
                                    sahi_pred = ObjectPrediction(
                                        bbox=[x1, y1, x2 - x1, y2 - y1],  # xywh
                                        category_id=cls_id,  # NEW: Set extracted class_id
                                        category_name="object",  # Adjust if classes known
                                        score=conf,
                                        segmentation=coco_segmentation,
                                        shift_amount=[0, 0],
                                        full_shape=[roi_height, roi_width]
                                    )
                                    all_predictions.append(sahi_pred)
                                except ValueError as ve:
                                    logger.warning(f"Skipping invalid prediction: {ve}")
                                    continue

                except RuntimeError as e:  # e.g., OOM
                    if "out of memory" in str(e):
                        logger.warning(f"OOM error: {e}; reducing batch size")
                        self.batch_size = max(1, self.batch_size // 2)
                        return self.infer_on_roi(image, roi)  # Retry with smaller batch
                    else:
                        raise

            # Phase 5: SAHI Postprocess (faster)
            logger.info(f"Running NMS post processing")
            postprocessor = NMSPostprocess(match_threshold=self.postprocess_threshold, class_agnostic=True)
            logger.info(f"Merging predictions")
            merged_preds = postprocessor(all_predictions)

            # Phase 6: Convert to Segmentations and offset to full image (parallelized)
            logger.info(f"Converting {len(merged_preds)} predictions to segmentations in parallel and offsetting to full image")

            # Use Pool for parallelism; adjust processes based on CPU (leave 1 core free)
            num_processes = max(1, mp.cpu_count() - 2)
            with mp.Pool(processes=num_processes) as pool:
                # Prepare args as list of tuples for map
                args = [(pred, roi.x1, roi.y1) for pred in merged_preds]
                detections = pool.starmap(process_pred, args)  # starmap unpacks tuples

            logger.info(f"Processed {len(detections)} detections in parallel")

            return detections

        except Exception as e:
            logger.error(f"Batching failed: {e}; falling back to sequential")
            import traceback
            traceback.print_exc()
            return self._sequential_inference(roi_image, roi)

    def _sequential_inference(self, roi_image: np.ndarray, roi: BoundingBox) -> List[Segmentation]:
        """Fallback to original sequential SAHI inference"""
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model=self.model,
            confidence_threshold=0.25,
            device=self.device
        )
        result = get_sliced_prediction(
            roi_image,
            detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            postprocess_type='NMS',
            postprocess_match_threshold=self.postprocess_threshold,
            postprocess_class_agnostic=True,
            verbose=0
        )
        detections = []
        for obj in result.object_prediction_list:
            if obj.mask is None or not obj.mask.bool_mask.any():  # Skip invalid masks
                logger.warning("Skipping prediction with invalid or empty mask")
                continue
            # NEW: Set class_id from obj (if available, else 0)
            obj.category_id = obj.category.id if hasattr(obj.category, 'id') else 0
            det = sahi_result_to_segmentation(obj)
            det.bbox = (det.bbox[0] + roi.x1, det.bbox[1] + roi.y1, det.bbox[2], det.bbox[3])
            detections.append(det)
        return detections