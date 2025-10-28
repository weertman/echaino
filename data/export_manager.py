# FILE: data/export_manager.py

import json
import csv
from typing import Dict
from data.models import ProjectData, Segmentation
import logging
from core.measurement_engine import MeasurementEngine

logger = logging.getLogger(__name__)


class ExportManager:
    """Handles exporting project data"""

    def __init__(self, config: dict):
        self.config = config

    def export_coco(self, project: ProjectData, file_path: str):
        # (unchanged)
        ...

    def export_csv(self, project: ProjectData, file_path: str):
        """Export measurements to CSV with deferred scaling if needed, including class_id and class_name"""
        include_confidence = self.config['export']['include_confidence']
        include_source = self.config['export'].get('include_source', False)

        # Deferred compute: ensure NEW fields are filled; we no longer care about diameter_um
        if project.scale_px_per_cm:
            engine = MeasurementEngine(project.scale_px_per_cm)
            needs_calc = any(
                seg.area_cm2 is None
                or seg.perimeter_cm is None
                or getattr(seg, 'diameter_best_fit_um', None) is None
                or getattr(seg, 'diameter_feret_max_um', None) is None
                for seg in project.segmentations
            )
            if needs_calc:
                logger.info("Deferred scaling: Computing measurements before CSV export")
                engine.batch_calculate(project.segmentations)

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.config['export']['csv_delimiter'])

            # NEW: no legacy diameter_um column
            header = [
                'id', 'box_id',
                'area_pixels', 'area_cm2',
                'perimeter_pixels', 'perimeter_cm',
                'diameter_best_fit_um',        # explicit best-fit
                'diameter_feret_max_um',       # explicit feret
                'class_id', 'class_name'
            ]
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
                    (seg.diameter_best_fit_um if seg.diameter_best_fit_um is not None else ''),
                    (seg.diameter_feret_max_um if seg.diameter_feret_max_um is not None else ''),
                    seg.class_id,
                    seg.class_name if seg.class_name else ''
                ]
                if include_confidence:
                    row.append(seg.confidence)
                if include_source:
                    row.append(seg.source)
                writer.writerow(row)

        logger.info(f"Exported CSV to {file_path}")
