# FILE: data\export_manager.py
# PATH: D:\urchinScanner\data\export_manager.py

import json
import csv
from typing import Dict
from data.models import ProjectData, Segmentation
import logging
from core.measurement_engine import MeasurementEngine  # NEW: Import for deferred calculation

logger = logging.getLogger(__name__)


class ExportManager:
    """Handles exporting project data"""

    def __init__(self, config: dict):
        self.config = config

    def export_coco(self, project: ProjectData, file_path: str):
        """Export to COCO format with multi-class support"""
        # NEW: Get classes from metadata or config (fallback to single-class)
        classes = project.metadata.get('classes', self.config.get('classes', ['urchin']))  # Fallback to ['urchin']
        categories = [
            {"id": i + 1, "name": name}  # COCO category_id starts at 1
            for i, name in enumerate(classes)
        ]

        coco_data = {
            "images": [{
                "id": 1,
                "width": project.image_shape[1],
                "height": project.image_shape[0],
                "file_name": project.image_path
            }],
            "categories": categories,  # NEW: Multi-class categories
            "annotations": []
        }

        for i, seg in enumerate(project.segmentations):
            if seg.polygon:
                # Flatten polygon for COCO segmentation
                coco_seg = [coord for point in seg.polygon for coord in point]
                annotation = {
                    "id": i + 1,
                    "image_id": 1,
                    "category_id": seg.class_id + 1 if hasattr(seg, 'class_id') else 1,  # NEW: Use seg.class_id + 1 (COCO starts at 1); default 1 if 0
                    "segmentation": [coco_seg],
                    "area": seg.area_pixels,
                    "bbox": list(seg.bbox),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)

        with open(file_path, 'w') as f:
            json.dump(coco_data, f, indent=4)

        logger.info(f"Exported COCO to {file_path}")

    def export_csv(self, project: ProjectData, file_path: str):
        """Export measurements to CSV with deferred scaling if needed, including class_id and class_name"""
        include_confidence = self.config['export']['include_confidence']
        include_source = self.config['export'].get('include_source', False)  # NEW

        # NEW: Deferred measurement calculation if scale available and not computed
        if project.scale_px_per_cm:
            engine = MeasurementEngine(project.scale_px_per_cm)
            needs_calc = any(seg.area_cm2 is None or seg.perimeter_cm is None or seg.diameter_um is None for seg in project.segmentations)
            if needs_calc:
                logger.info("Deferred scaling: Computing measurements before CSV export")
                engine.batch_calculate(project.segmentations)

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.config['export']['csv_delimiter'])

            # Write header with optional columns and NEW: diameter_um, class_id, class_name
            header = ['id', 'box_id', 'area_pixels', 'area_cm2', 'perimeter_pixels', 'perimeter_cm', 'diameter_um', 'class_id', 'class_name']
            if include_confidence:
                header.append('confidence')
            if include_source:
                header.append('source')
            writer.writerow(header)

            for seg in project.segmentations:
                row = [
                    seg.id,
                    seg.box_id,
                    seg.area_pixels,
                    seg.area_cm2 if seg.area_cm2 is not None else '',
                    seg.perimeter_pixels,
                    seg.perimeter_cm if seg.perimeter_cm is not None else '',
                    seg.diameter_um if seg.diameter_um is not None else '',  # NEW: Add diameter_um
                    seg.class_id,  # NEW: class_id (default 0)
                    seg.class_name if seg.class_name else ''  # NEW: class_name (optional)
                ]
                if include_confidence:
                    row.append(seg.confidence)
                if include_source:
                    row.append(seg.source)
                writer.writerow(row)

        logger.info(f"Exported CSV to {file_path}")