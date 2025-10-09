# FILE: data\project_manager.py
# PATH: D:\urchinScanner\data\project_manager.py

import json
from dataclasses import asdict, fields  # Added fields for filtering
from typing import Optional, List
from data.models import ProjectData, BoundingBox, Segmentation, CalibrationLine, Tile
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProjectManager:
    """Handles project saving and loading"""

    def save_project(self, project: ProjectData, file_path: str):
        """Save project to JSON"""
        data = asdict(project)

        # Convert numpy arrays if any
        for seg in data['segmentations']:
            if seg['mask'] is not None:
                seg['mask'] = seg['mask'].tolist()  # Convert to list if keeping masks

        # NEW: Ensure classes in metadata (even if empty)
        if 'classes' not in data['metadata']:
            data['metadata']['classes'] = []

        # Ensure new fields are serialized
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info(f"Saved project to {file_path}")

    def load_project(self, file_path: str) -> Optional[ProjectData]:
        """Load project from JSON"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Enhanced detection and assignment for old files
            metadata = data.get('metadata', {})
            is_legacy = 'classes' not in metadata or not any('class_id' in b for b in data.get('boxes', []) + data.get('all_boxes', []) + data.get('segmentations', []))

            if is_legacy:
                logger.info("Detected legacy single-class project; assuming class 'urchin'")
                metadata['classes'] = ["urchin"]  # Set to single-class "urchin"
                metadata['is_legacy'] = True  # NEW: Flag for UI to trigger dialog

            # Parse boxes with source migration and NEW: class_id
            boxes = []
            for box_data in data.get('boxes', []):
                box = BoundingBox(
                    id=box_data['id'],
                    x1=box_data['x1'],
                    y1=box_data['y1'],
                    x2=box_data['x2'],
                    y2=box_data['y2'],
                    status=box_data.get('status', 'pending'),
                    class_id=box_data.get('class_id', 0)  # NEW: Default 0 if missing
                )
                box.source = box_data.get('source', 'manual')  # New: Migrate if missing
                if is_legacy:
                    box.class_name = "urchin"  # NEW: Set class_name for consistency
                boxes.append(box)

            # Similar for all_boxes (if separate in data)
            all_boxes = []
            for b_data in data.get('all_boxes', []):
                box = BoundingBox(
                    id=b_data['id'],
                    x1=b_data['x1'],
                    y1=b_data['y1'],
                    x2=b_data['x2'],
                    y2=b_data['y2'],
                    status=b_data.get('status', 'pending'),
                    class_id=b_data.get('class_id', 0)  # NEW: Default 0 if missing
                )
                box.source = b_data.get('source', 'manual')  # New: Migrate if missing
                if is_legacy:
                    box.class_name = "urchin"  # NEW: Set class_name for consistency
                all_boxes.append(box)

            # If all_boxes is empty but boxes exist (old format)
            if not all_boxes and boxes:
                all_boxes = [BoundingBox(**{k: v for k, v in asdict(b).items() if k in {f.name for f in fields(BoundingBox)}}) for b in boxes]  # Copy

            # Reconstruct segmentations with filtering and NEW: class_id, class_name
            seg_fields = {f.name for f in fields(Segmentation)}
            segmentations = []
            for s_data in data.get('segmentations', []):
                if not isinstance(s_data, dict):
                    logger.error(f"Skipping invalid segmentation data type: {type(s_data)}")
                    continue
                mask = None
                if 'mask' in s_data and s_data['mask'] is not None:
                    mask = np.array(s_data['mask'], dtype=bool)  # Assume bool mask

                # Use filtered dict for construction
                filtered_data = {k: v for k, v in s_data.items() if k in seg_fields}
                # NEW: Ensure class_id and class_name with defaults
                filtered_data['class_id'] = s_data.get('class_id', 0)
                filtered_data['class_name'] = s_data.get('class_name', None)
                seg = Segmentation(**filtered_data)
                seg.source = s_data.get('source', 'manual')  # New: Migrate if missing
                if is_legacy:
                    seg.class_name = "urchin"  # NEW: Set class_name for consistency
                segmentations.append(seg)

            # Reconstruct calibration if present
            cal_data = data.get('calibration_line')
            calibration_line = None
            if cal_data:
                # In load_project, when creating calibration_line:
                calibration_line = CalibrationLine(
                    start_x=cal_data['start_x'],
                    start_y=cal_data['start_y'],
                    end_x=cal_data['end_x'],
                    end_y=cal_data['end_y'],
                    real_distance=cal_data['real_distance'],
                    unit=cal_data.get('unit', 'cm')  # Backward compat
                )

            # NEW: Ensure metadata has 'classes' with default empty list
            if 'classes' not in metadata:
                metadata['classes'] = []

            project = ProjectData(
                image_path=data['image_path'],
                image_shape=tuple(data['image_shape']),
                boxes=boxes,
                all_boxes=all_boxes,
                segmentations=segmentations,
                scale_px_per_cm=data.get('scale_px_per_cm'),
                calibration_line=calibration_line,
                metadata=metadata,  # Updated with classes
                archive_path=data.get('archive_path')
            )

            if any(b.source is None for b in boxes) or any(s.source is None for s in segmentations):
                logger.info("Migrated old project: set default 'manual' source for annotations")

            logger.info(f"Loaded project from {file_path}")
            return project

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            return None