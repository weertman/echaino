# FILE: data/project_manager.py
# PATH: D:\\echaino\\data\\project_manager.py

import json
from dataclasses import asdict, fields  # fields used to whitelist dataclass attrs
from typing import Optional, List
from data.models import ProjectData, BoundingBox, Segmentation, CalibrationLine, Tile
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProjectManager:
    """Handles project saving and loading (with legacy compatibility)."""

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    def save_project(self, project: ProjectData, file_path: str):
        """
        Save project to JSON.

        Notes:
          - numpy arrays (e.g., masks) are converted to lists if present.
          - metadata['classes'] is ensured for multi-class compatibility.
        """
        data = asdict(project)

        # Convert numpy arrays in segmentations to JSON-friendly lists if present
        for seg in data.get('segmentations', []):
            if seg.get('mask') is not None:
                # seg['mask'] is a numpy array in memory; cast to list for JSON
                try:
                    seg['mask'] = np.asarray(seg['mask']).astype(bool).tolist()
                except Exception:
                    # Be defensive: if mask is already a list or fails conversion, leave as-is
                    pass

        # Ensure classes in metadata (even if empty)
        data.setdefault('metadata', {})
        if 'classes' not in data['metadata']:
            data['metadata']['classes'] = []

        # Write JSON
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info(f"Saved project to {file_path}")

    # ---------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------
    def load_project(self, file_path: str) -> Optional[ProjectData]:
        """
        Load project from JSON, handling:
          - Legacy single-class projects (injects classes=['urchin'])
          - Missing 'source' on boxes/segmentations (defaults to 'manual')
          - New diameter fields migration:
              * if only `diameter_um` exists, copy it to `diameter_best_fit_um`
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            # Legacy detection: no classes declared and/or no class_id fields anywhere
            is_legacy = (
                'classes' not in metadata
                or not any(
                    ('class_id' in b)
                    for b in data.get('boxes', [])
                    + data.get('all_boxes', [])
                    + data.get('segmentations', [])
                )
            )

            if is_legacy:
                logger.info("Detected legacy single-class project; assuming class 'urchin'")
                metadata['classes'] = ["urchin"]
                metadata['is_legacy'] = True  # for downstream UI hinting

            # ------------------------------
            # Rebuild boxes
            # ------------------------------
            boxes: List[BoundingBox] = []
            for box_data in data.get('boxes', []):
                box = BoundingBox(
                    id=box_data['id'],
                    x1=box_data['x1'],
                    y1=box_data['y1'],
                    x2=box_data['x2'],
                    y2=box_data['y2'],
                    status=box_data.get('status', 'pending'),
                    class_id=box_data.get('class_id', 0)  # default 0
                )
                # Migrate 'source' if absent
                box.source = box_data.get('source', 'manual')
                if is_legacy:
                    # convenient for UI (non-dataclass attribute is fine)
                    box.class_name = "urchin"
                boxes.append(box)

            # Rebuild all_boxes (copy from boxes if absent in legacy files)
            all_boxes: List[BoundingBox] = []
            for b_data in data.get('all_boxes', []):
                box = BoundingBox(
                    id=b_data['id'],
                    x1=b_data['x1'],
                    y1=b_data['y1'],
                    x2=b_data['x2'],
                    y2=b_data['y2'],
                    status=b_data.get('status', 'pending'),
                    class_id=b_data.get('class_id', 0)
                )
                box.source = b_data.get('source', 'manual')
                if is_legacy:
                    box.class_name = "urchin"
                all_boxes.append(box)

            # If all_boxes not present but boxes exist, create a shallow copy structure
            if not all_boxes and boxes:
                # copy only declared dataclass fields
                bbox_fields = {f.name for f in fields(BoundingBox)}
                all_boxes = [BoundingBox(**{k: v for k, v in asdict(b).items() if k in bbox_fields}) for b in boxes]

            # ------------------------------
            # Rebuild segmentations
            # ------------------------------
            seg_fields = {f.name for f in fields(Segmentation)}
            segmentations: List[Segmentation] = []

            for s_data in data.get('segmentations', []):
                if not isinstance(s_data, dict):
                    logger.error(f"Skipping invalid segmentation data type: {type(s_data)}")
                    continue

                # Rehydrate mask if present
                mask = None
                if 'mask' in s_data and s_data['mask'] is not None:
                    try:
                        mask = np.array(s_data['mask'], dtype=bool)
                    except Exception:
                        mask = None

                # Filter allowed fields for dataclass constructor
                filtered_data = {k: v for k, v in s_data.items() if k in seg_fields}

                # Ensure class fields exist
                filtered_data['class_id'] = s_data.get('class_id', 0)
                filtered_data['class_name'] = s_data.get('class_name', None)

                # Insert mask (may be None)
                filtered_data['mask'] = mask

                # Construct Segmentation
                seg = Segmentation(**filtered_data)

                # Migrate 'source' if absent
                seg.source = s_data.get('source', 'manual')

                # ------------------------------
                # NEW: Diameter migration shim
                # If a legacy file only has diameter_um, mirror it into diameter_best_fit_um.
                # Keep diameter_um as the legacy alias (MeasurementEngine will keep them in sync going forward).
                # ------------------------------
                try:
                    has_best_fit = getattr(seg, 'diameter_best_fit_um', None) is not None
                    has_legacy   = getattr(seg, 'diameter_um', None) is not None
                    if not has_best_fit and has_legacy:
                        seg.diameter_best_fit_um = seg.diameter_um
                except Exception as e:
                    logger.debug(f"Diameter migration skipped for seg {getattr(seg, 'id', '?')}: {e}")

                # For legacy projects, add class_name convenience
                if is_legacy and getattr(seg, 'class_name', None) is None:
                    seg.class_name = "urchin"

                segmentations.append(seg)

            # ------------------------------
            # Rebuild calibration (if present)
            # ------------------------------
            cal_data = data.get('calibration_line')
            calibration_line = None
            if cal_data:
                calibration_line = CalibrationLine(
                    start_x=cal_data['start_x'],
                    start_y=cal_data['start_y'],
                    end_x=cal_data['end_x'],
                    end_y=cal_data['end_y'],
                    real_distance=cal_data['real_distance'],
                    unit=cal_data.get('unit', 'cm')
                )

            # Ensure metadata has classes
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
                metadata=metadata,
                archive_path=data.get('archive_path')
            )

            # Log source migration info if any missing
            if any(getattr(b, 'source', None) is None for b in boxes) or any(getattr(s, 'source', None) is None for s in segmentations):
                logger.info("Migrated old project: set default 'manual' source for annotations")

            logger.info(f"Loaded project from {file_path}")
            return project

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            return None
