# FILE: ui\control_panel.py
# PATH: D:\urchinScanner\ui\control_panel.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox, QCheckBox, QComboBox,
    QLineEdit, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
import os
import random
import colorsys
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Use the dedicated dialog from its own module (single source of truth)
from ui.manage_classes_dialog import ManageClassesDialog


def _rgb_tuple_to_list(rgb: Tuple[int, int, int]) -> List[int]:
    return [int(max(0, min(255, c))) for c in rgb]


def _distinct_color(existing_rgb: List[List[int]]) -> List[int]:
    """
    Pick a random-ish vivid color (RGB 0-255) that is not too close to any in existing_rgb.
    Uses HSV sampling and simple distance thresholding in RGB space.
    """
    used = [tuple(c) for c in existing_rgb] if existing_rgb else []
    # Avoid near-black/near-white; aim for vivid colors
    def sample_candidate() -> Tuple[int, int, int]:
        h = random.random()            # 0..1
        s = random.uniform(0.55, 0.95) # vivid
        v = random.uniform(0.75, 0.95) # bright
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def dist2(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
        return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

    # Threshold tuned to keep colors visually distinct (≈ Euclidean distance > ~140)
    THRESH2 = 140**2

    for _ in range(200):
        cand = sample_candidate()
        if all(dist2(cand, u) > THRESH2 for u in used):
            return _rgb_tuple_to_list(cand)

    # Fallback: just return the best among 50 trials
    best = None
    best_min_d2 = -1
    for _ in range(50):
        cand = sample_candidate()
        if not used:
            return _rgb_tuple_to_list(cand)
        mind2 = min((dist2(cand, u) for u in used))
        if mind2 > best_min_d2:
            best_min_d2 = mind2
            best = cand
    return _rgb_tuple_to_list(best or (0, 160, 255))


def _reconcile_class_colors(old_names: List[str],
                            old_colors: List[List[int]],
                            new_names: List[str]) -> List[List[int]]:
    """
    Build a color list aligned to new_names:
      - preserve colors for names that existed before
      - assign a new distinct color for brand-new names
    """
    name_to_color: Dict[str, List[int]] = {}
    for i, nm in enumerate(old_names or []):
        if i < len(old_colors) and isinstance(old_colors[i], (list, tuple)) and len(old_colors[i]) == 3:
            name_to_color[nm] = [int(c) for c in old_colors[i]]
    new_colors: List[List[int]] = []
    for nm in new_names:
        if nm in name_to_color:
            new_colors.append(name_to_color[nm])
        else:
            new_colors.append(_distinct_color(new_colors or list(name_to_color.values())))
    return new_colors


class ControlPanel(QWidget):
    """Right-side control panel"""

    process_clicked = pyqtSignal()
    classes_updated = pyqtSignal(list)          # emits list[str] (class names)
    export_coco_clicked = pyqtSignal()
    export_csv_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    redo_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    scale_changed = pyqtSignal(float)
    calibration_clicked = pyqtSignal()
    tile_size_changed = pyqtSignal(int)
    debug_mode_toggled = pyqtSignal(bool)
    show_debug_clicked = pyqtSignal()
    show_boxes_toggled = pyqtSignal(bool)
    show_segmentations_toggled = pyqtSignal(bool)
    show_grid_toggled = pyqtSignal(bool)
    select_roi_clicked = pyqtSignal()
    run_yolo_clicked = pyqtSignal()
    edit_mode_toggled = pyqtSignal(bool)
    reprocess_clicked = pyqtSignal()
    class_changed = pyqtSignal(int)

    def __init__(self, config: dict):
        super().__init__()
        # Ensure config is always a dict to avoid NoneAttribute surprises
        self.config = config if isinstance(config, dict) else {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.box_count_label = QLabel("Boxes: 0")
        self.seg_count_label = QLabel("Segmentations: 0")
        self.calibration_label = QLabel("Scale: Not set")
        stats_layout.addWidget(self.box_count_label)
        stats_layout.addWidget(self.seg_count_label)
        stats_layout.addWidget(self.calibration_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        self.process_btn = QPushButton("Process Annotations")
        self.process_btn.clicked.connect(self.process_clicked.emit)
        controls_layout.addWidget(self.process_btn)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_clicked.emit)
        controls_layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo_clicked.emit)
        controls_layout.addWidget(self.redo_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        controls_layout.addWidget(self.clear_btn)

        # Class selection + manage
        class_layout = QHBoxLayout()
        self.class_combo = QComboBox()
        self._refresh_class_combo(self.config.get('classes', ['Class 0']))
        self.class_combo.currentIndexChanged.connect(self.class_changed.emit)
        class_layout.addWidget(QLabel("Select Class:"))
        class_layout.addWidget(self.class_combo)

        manage_btn = QPushButton("Manage Classes")
        manage_btn.clicked.connect(self.manage_classes)
        class_layout.addWidget(manage_btn)
        controls_layout.addLayout(class_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Calibration
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout()
        self.calibrate_btn = QPushButton("Calibrate Scale")
        self.calibrate_btn.clicked.connect(self.calibration_clicked.emit)
        cal_layout.addWidget(self.calibrate_btn)

        self.scale_input = QLineEdit()
        self.scale_input.setPlaceholderText("Manual scale (px/cm)")
        self.scale_input.returnPressed.connect(self._emit_scale_changed)
        cal_layout.addWidget(self.scale_input)

        cal_group.setLayout(cal_layout)
        layout.addWidget(cal_group)

        # Tiling
        tiling_group = QGroupBox("Tiling")
        tiling_layout = QVBoxLayout()
        self.tile_size_combo = QComboBox()
        self.tile_size_combo.addItems(["512", "640", "768", "1024"])
        self.tile_size_combo.setCurrentText(str(self.config.get('tiling', {}).get('base_tile_size', 640)))
        self.tile_size_combo.currentTextChanged.connect(lambda t: self.tile_size_changed.emit(int(t)))
        tiling_layout.addWidget(QLabel("Tile Size:"))
        tiling_layout.addWidget(self.tile_size_combo)

        self.show_grid_check = QCheckBox("Show Tile Grid")
        self.show_grid_check.toggled.connect(self.show_grid_toggled.emit)
        tiling_layout.addWidget(self.show_grid_check)

        tiling_group.setLayout(tiling_layout)
        layout.addWidget(tiling_group)

        # Display
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()
        self.show_boxes_check = QCheckBox("Show Boxes")
        self.show_boxes_check.setChecked(True)
        self.show_boxes_check.toggled.connect(self.show_boxes_toggled.emit)
        display_layout.addWidget(self.show_boxes_check)

        self.show_segmentations_check = QCheckBox("Show Segmentations")
        self.show_segmentations_check.setChecked(True)
        self.show_segmentations_check.toggled.connect(self.show_segmentations_toggled.emit)
        display_layout.addWidget(self.show_segmentations_check)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        self.export_coco_btn = QPushButton("Export COCO JSON")
        self.export_coco_btn.clicked.connect(self.export_coco_clicked.emit)
        export_layout.addWidget(self.export_coco_btn)

        self.export_csv_btn = QPushButton("Export CSV Measurements")
        self.export_csv_btn.clicked.connect(self.export_csv_clicked.emit)
        export_layout.addWidget(self.export_csv_btn)

        self.generate_report_check = QCheckBox("Generate Visualization Report")
        self.generate_report_check.setChecked(True)
        export_layout.addWidget(self.generate_report_check)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Debug
        debug_group = QGroupBox("Debug")
        debug_layout = QVBoxLayout()
        self.debug_mode_check = QCheckBox("Debug Mode")
        self.debug_mode_check.setChecked(self.config.get('debug', False))
        self.debug_mode_check.toggled.connect(self.debug_mode_toggled.emit)
        debug_layout.addWidget(self.debug_mode_check)

        self.show_debug_btn = QPushButton("Show Tile Debug Viewer")
        self.show_debug_btn.clicked.connect(self.show_debug_clicked.emit)
        debug_layout.addWidget(self.show_debug_btn)

        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)

        # YOLO
        yolo_group = QGroupBox("YOLO Integration")
        yolo_layout = QVBoxLayout()
        self.select_roi_btn = QPushButton("Select ROI")
        self.select_roi_btn.clicked.connect(self.select_roi_clicked.emit)
        yolo_layout.addWidget(self.select_roi_btn)

        model_path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to trained YOLO model (.pt)")
        model_path_layout.addWidget(self.model_path_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_model)
        model_path_layout.addWidget(browse_btn)
        yolo_layout.addLayout(model_path_layout)

        self.run_yolo_btn = QPushButton("Run YOLO Detection")
        self.run_yolo_btn.clicked.connect(self.run_yolo_clicked.emit)
        yolo_layout.addWidget(self.run_yolo_btn)

        yolo_group.setLayout(yolo_layout)
        layout.addWidget(yolo_group)

        # Edit mode
        self.edit_mode_check = QCheckBox("Edit Mode (Shift+E)")
        self.edit_mode_check.setToolTip("Toggle with Shift+E hotkey")
        self.edit_mode_check.toggled.connect(self.edit_mode_toggled.emit)
        self.edit_mode_check.toggled.connect(self._toggle_controls)
        layout.addWidget(self.edit_mode_check)

        # Reprocess edited
        self.reprocess_btn = QPushButton("Re-process Edited")
        self.reprocess_btn.clicked.connect(self.reprocess_clicked.emit)
        layout.addWidget(self.reprocess_btn)

        layout.addStretch()

    def _emit_scale_changed(self):
        try:
            scale = float(self.scale_input.text())
            self.scale_changed.emit(scale)
        except ValueError:
            pass

    def update_statistics(self, num_boxes: int, num_segmentations: int):
        self.box_count_label.setText(f"Boxes: {num_boxes}")
        self.seg_count_label.setText(f"Segmentations: {num_segmentations}")

    def get_scale(self) -> Optional[float]:
        if hasattr(self, '_calibration_scale') and self._calibration_scale:
            return self._calibration_scale
        if self.scale_input.text():
            try:
                return float(self.scale_input.text())
            except ValueError:
                pass
        return self.config.get('default_scale_px_per_cm')

    def update_calibration_status(self, calibration_line):
        if calibration_line:
            self.calibration_label.setText(f"Scale: {calibration_line.pixels_per_unit:.2f} px/{calibration_line.unit}")
            self._calibration_scale = calibration_line.pixels_per_unit
            self.scale_input.setText(str(calibration_line.pixels_per_unit))
        else:
            self.calibration_label.setText("Scale: Not set")
            self._calibration_scale = None

    def get_tile_size(self) -> int:
        return int(self.tile_size_combo.currentText())

    def get_model_path(self):
        path = self.model_path_input.text().strip()
        if not path:
            QMessageBox.warning(self, "Error", "Please select a model path")
            return ""
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Model file not found")
            return ""
        return path

    def browse_model(self):
        models_dir = self.config.get('yolo', {}).get('models_dir', './models/')
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Trained YOLO Model",
            models_dir,
            "YOLO Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def _refresh_class_combo(self, class_names: List[str]):
        """Populate the class combo and paint items using config['ui']['class_colors'] if present."""
        self.class_combo.clear()
        class_names = class_names or ['Class 0']
        self.class_combo.addItems(class_names)

        colors = self.config.get('ui', {}).get('class_colors', [])
        for i, nm in enumerate(class_names):
            if i < len(colors) and isinstance(colors[i], (list, tuple)) and len(colors[i]) == 3:
                r, g, b = colors[i]
                self.class_combo.setItemData(i, QColor(int(r), int(g), int(b)), role=9)  # BackgroundRole
                # Also set ForegroundRole for contrast if background is dark
                luminance = 0.2126*r + 0.7152*g + 0.0722*b
                fg = QColor(255, 255, 255) if luminance < 140 else QColor(0, 0, 0)
                self.class_combo.setItemData(i, fg, role=6)  # ForegroundRole

    def manage_classes(self):
        """
        Open the dedicated ManageClassesDialog, update config + combo,
        assign distinct colors to any new classes, and emit classes_updated.
        """
        current_names = self.config.get('classes', [])
        current_colors = self.config.get('ui', {}).get('class_colors', [])

        dlg = ManageClassesDialog(current_names, parent=self)
        if dlg.exec():
            new_names = dlg.classes() or []

            # Reconcile colors by class name
            new_colors = _reconcile_class_colors(current_names, current_colors, new_names)

            # Update config in-memory
            self.config['classes'] = list(new_names)
            self.config.setdefault('ui', {})
            self.config['ui']['class_colors'] = new_colors

            # Refresh UI (colored combo)
            self._refresh_class_combo(new_names)

            # Emit once
            try:
                self.classes_updated.emit(new_names)
            except Exception as e:
                logger.error(f"Failed to emit classes_updated: {e}")

    def _toggle_controls(self, checked: bool):
        disable = checked
        for w in (
            self.process_btn,
            self.select_roi_btn,
            self.calibrate_btn,
            self.undo_btn,
            self.redo_btn,
            self.clear_btn,
            self.export_coco_btn,
            self.export_csv_btn,
        ):
            w.setEnabled(not disable)

    def get_generate_report(self) -> bool:
        return self.generate_report_check.isChecked()
