# FILE: ui\main_window.py
# PATH: D:\\echaino\\ui\main_window.py

import sys
import yaml
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QMenuBar, QMenu, QMessageBox, QFileDialog,
                             QProgressDialog, QSplitter, QInputDialog)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QThread, QObject, pyqtSignal, QMutex
from PyQt6.QtGui import QAction, QKeySequence, QKeyEvent

import numpy as np
import torch
import os  # Added for absolute paths
from datetime import datetime

from ui.image_canvas import ImageCanvas
from ui.control_panel import ControlPanel
from ui.debug_tile_viewer import DebugTileViewer
from core.annotation_manager import AnnotationManager
from data.export_manager import ExportManager
from data.project_manager import ProjectManager
from data.archive_manager import ArchiveManager
from data.models import BoundingBox, Segmentation, CalibrationLine
from utils.image_utils import MemoryEfficientImageLoader
from core.annotation_manager import EditBoxCommand, DeleteBoxCommand
from core.yolo_trainer import YoloTrainer
from core.yolo_inferencer import YoloInferencer
from ui.yolo_train_dialog import YoloTrainDialog
from ui.cross_archive_analysis_dialog import CrossArchiveAnalysisDialog  # New import
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Thread for processing annotations"""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, annotation_manager, image, scale):
        super().__init__()
        self.annotation_manager = annotation_manager
        self.image = image
        self.scale = scale

    def run(self):
        try:
            segmentations = self.annotation_manager.process_annotations(
                self.image,
                self.scale,
                lambda i, t, m: self.progress.emit(i, t, m)
            )
            self.finished.emit(segmentations)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.error.emit(str(e))


# NEW: Worker classes (similar to ProcessingThread)
class YoloTrainWorker(QObject):
    finished = pyqtSignal(str)  # Model path
    error = pyqtSignal(str)

    def __init__(self, trainer, archives, params):
        super().__init__()
        self.trainer = trainer
        self.archives = archives
        self.params = params

    def run(self):
        try:
            logger.info("Starting YOLO training preparation")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
            dataset_path = self.trainer.prepare_dataset(self.archives, self.params)
            model_path = self.trainer.train_model(dataset_path, self.params, timestamp)  # Pass timestamp
            self.finished.emit(str(model_path))
        except Exception as e:
            import traceback
            err_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            logger.error(err_msg)
            self.error.emit(err_msg)

class YoloInferenceWorker(QObject):
    detections_ready = pyqtSignal(list)
    finished = pyqtSignal()  # New: Explicit finished signal after detections
    error = pyqtSignal(str)  # New: For exceptions

    def __init__(self, inferencer, image, roi, model_path):
        super().__init__()
        self.inferencer = inferencer
        self.image = image
        self.roi = roi
        self.model_path = model_path

    def run(self):
        try:
            logger.info("Starting YOLO inference")
            self.inferencer.load_model(self.model_path)
            detections = self.inferencer.infer_on_roi(self.image, self.roi)  # Updated call (removed {} if not needed)
            self.detections_ready.emit(detections)
            self.finished.emit()  # Emit finished after ready
        except Exception as e:
            import traceback
            err_msg = f"Inference error: {str(e)}\n{traceback.format_exc()}"
            logger.error(err_msg)
            self.error.emit(err_msg)

class CrossArchiveAnalysisWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # Emit report_path
    error = pyqtSignal(str)

    def __init__(self, archive_manager, project_manager, export_manager, archives, params, config):
        super().__init__()
        self.archive_manager = archive_manager
        self.project_manager = project_manager
        self.export_manager = export_manager
        self.archives = archives
        self.params = params
        self.config = config

    def run(self):
        try:
            from utils.cross_archive_analyzer import CrossArchiveAnalyzer
            import pandas as pd
            import os
            import gc

            dfs = []
            selected_class_index = self.params.get('selected_class_index', 0)
            selected_class_name = self.params.get('selected_class_name', '')

            for i, archive in enumerate(self.archives):
                self.progress.emit(i)
                csv_path = os.path.join(self.config['archive']['root_dir'], archive, "measurements.csv")
                if not os.path.exists(csv_path):
                    if self.params.get('auto_generate', True):
                        try:
                            latest_json = self.archive_manager.get_latest_json(archive)
                            if not latest_json:
                                raise FileNotFoundError("No JSON found")
                            project = self.project_manager.load_project(latest_json)
                            self.export_manager.export_csv(project, csv_path)
                            logger.info(f"Generated CSV for {archive}")
                        except Exception as e:
                            logger.error(f"Failed to generate CSV for {archive}: {e}")
                            self.error.emit(f"Failed to generate CSV for {archive}: {str(e)}")
                            continue
                    else:
                        continue

                try:
                    df = pd.read_csv(csv_path)
                    df['archive'] = archive

                    # Filter by selected class if class_id column exists
                    if 'class_id' in df.columns and selected_class_index is not None:
                        df = df[df['class_id'] == selected_class_index]
                        logger.info(
                            f"Filtered {archive} to class {selected_class_index} ({selected_class_name}): {len(df)} rows")

                    # Standardize to µm
                    if 'area_cm2' in df.columns:
                        df['area_um2'] = df['area_cm2'] * 1e8
                    if 'perimeter_cm' in df.columns:
                        df['perimeter_um'] = df['perimeter_cm'] * 1e4
                    # Drop pixel cols if present
                    pixel_cols = [col for col in df.columns if 'pixels' in col]
                    if pixel_cols:
                        df.drop(columns=pixel_cols, inplace=True)
                    dfs.append(df)
                except Exception as e:
                    self.error.emit(f"Failed to load CSV for {archive}: {str(e)}")
                    continue

            if not dfs:
                raise ValueError("No valid data loaded from selected archives")

            merged_df = pd.concat(dfs, ignore_index=True)

            # Pass the selected class info to the analyzer
            analyzer = CrossArchiveAnalyzer(merged_df, self.params['metrics'], selected_class_name)
            report_path = analyzer.generate_report(self.config['archive']['root_dir'])

            gc.collect()
            self.finished.emit(report_path)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EchAIno Annotator")
        self.setGeometry(100, 100, 1400, 900)

        # Load configuration
        self.config = self._load_config()
        # Guard against None or invalid types
        if not isinstance(self.config, dict):
            self.config = {}
            try:
                self.config.update(self._load_config() or {})
            except Exception:
                pass

        # NEW: Classes for multi-class (from config)
        self.classes = self.config.get('classes', [])  # List[str], empty for single-class

        # Initialize components
        self.annotation_manager = AnnotationManager(self.config)  # Assumes AnnotationManager uses config with classes
        self.export_manager = ExportManager(self.config)
        self.project_manager = ProjectManager()
        self.archive_manager = ArchiveManager(self.config, self.project_manager)  # Added
        self.yolo_trainer = YoloTrainer(self.config, self.archive_manager)
        self.yolo_inferencer = YoloInferencer(self.config)  # NEW: Pass config for batch_size

        # Current project state
        self.current_image_path = None
        self.current_image = None
        self.image_loader = None
        self.calibration_line = None
        self.current_archive_path = None  # Added
        self.current_roi = None  # NEW: For inference

        # Auto-save timer
        self.auto_save_timer = QTimer(self)  # Added
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.start(self.config['archive']['auto_save_interval_sec'] * 1000)

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

        # Debug viewer
        self.debug_viewer = None

        # Enable debug mode
        # self.config['debug'] = True

        # NEW: Mode lock and current mode
        self.mode_lock = QMutex()  # New: For thread safety
        self.current_mode = 'normal'  # New: Track mode

    def _load_config(self) -> dict:
        """Load configuration from yaml file; always returns a dict with sensible defaults."""
        # Safe, complete defaults used throughout app
        defaults = {
            'sam': {
                'checkpoint': 'sam_vit_h_4b8939.pth',
                'model_type': 'vit_h',
                'device': 'cpu',
            },
            'yolo': {
                'model': 'yolov8n.pt',
                'pretrained_model': 'yolov8n.pt',
                'imgsz': 640,
                'conf': 0.25,
                'iou': 0.45,
                'device': 'auto',
                'sahi_batch_size': 4,
                'models_dir': './models',
                'dataset_output_dir': './datasets',
                'train_epochs': 50,
                'batch_size': 16,
                'img_size': 640,
                'train_val_split': 0.8,
                'min_clip_area': 10,
            },
            'image': {
                'scale_factor': 1.0,
                'dpi': 300,
            },
            'archive': {
                'root_dir': 'archives',
                'subfolder_format': '{timestamp}_{image_name}',
                'auto_save_interval_sec': 60,
            },
            'tiling': {
                'base_tile_size': 640,
                'overlap_ratio': 0.2,
                # NEW: center-accept + halo (opt-in, low churn)
                'center_accept': True,              # enable the filtering
                'center_accept_halo_px': 96         # shrink accept window by this many px on each edge
            },
            'sahi': {
                'slice_size': 640,
                'overlap_ratio': 0.2,
                'postprocess_match_threshold': 0.5,
            },
            'ui': {
                'box_color': [255, 0, 0],
                'box_width': 2,
                'class_colors': [],
            },
            'processing': {
                'batch_size': 1,
                'enable_deduplication': False,
                'enable_merge_partials': True,
                'keep_full_masks': False,
            },
            'export': {
                'include_confidence': True,
                'include_source': True,
                'csv_delimiter': ',',
            },
            'classes': [],  # multi-class list
            'debug': False,
            'default_scale_px_per_cm': None,
        }

        import yaml
        from pathlib import Path
        cfg = {}
        config_path = Path("config.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                cfg = data if isinstance(data, dict) else {}
            except Exception as e:
                logger.warning(f"Failed to read config.yaml; using defaults: {e}")
                cfg = {}

        # Shallow merge (dict-within-dict)
        merged = dict(defaults)
        for k, v in (cfg or {}).items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v

        # Resolve yolo device if 'auto'
        device = merged.get('yolo', {}).get('device', 'auto')
        if device == 'auto':
            try:
                import torch  # safe local import
                merged['yolo']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                merged.setdefault('yolo', {})
                merged['yolo']['device'] = 'cpu'

        # Guarantee shape invariants used elsewhere
        merged.setdefault('classes', [])
        merged.setdefault('ui', {}).setdefault('class_colors', [])
        merged.setdefault('tiling', {}).setdefault('overlap_ratio', 0.2)

        return merged

    def _save_config(self):
        """Persist current configuration to config.yaml."""
        try:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, sort_keys=False)
        except Exception as e:
            try:
                logger.error(f'Failed to save configuration: {e}')
            except Exception:
                pass

    def _setup_ui(self):
        """Setup UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Create image canvas with annotation manager reference
        self.canvas = ImageCanvas(self.config, self.annotation_manager)
        splitter.addWidget(self.canvas)

        # Create control panel
        self.control_panel = ControlPanel(self.config)
        splitter.addWidget(self.control_panel)

        # Set splitter sizes
        splitter.setSizes([1000, 400])

    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        browse_archives_action = QAction("Browse Archives...", self)
        browse_archives_action.triggered.connect(self.browse_archives)
        file_menu.addAction(browse_archives_action)

        file_menu.addSeparator()

        save_project_action = QAction("Save Project...", self)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        load_project_action = QAction("Load Project...", self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(fit_action)

        view_menu.addSeparator()

        # Add calibration action
        calibration_action = QAction("Calibrate Scale", self)
        calibration_action.setShortcut("Ctrl+R")  # R for Ruler
        calibration_action.triggered.connect(self.start_calibration)
        view_menu.addAction(calibration_action)

        view_menu.addSeparator()

        # Add debug viewer action
        debug_action = QAction("Show Tile Debug Viewer", self)
        debug_action.setShortcut("Ctrl+D")
        debug_action.triggered.connect(self.show_debug_viewer)
        view_menu.addAction(debug_action)

        # Add toggle debug mode action
        self.debug_mode_action = QAction("Debug Mode", self)
        self.debug_mode_action.setCheckable(True)
        self.debug_mode_action.setChecked(self.config.get('debug', False))
        self.debug_mode_action.toggled.connect(self.toggle_debug_mode)
        view_menu.addAction(self.debug_mode_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # NEW: YOLO menu
        yolo_menu = menubar.addMenu("YOLO")
        train_action = QAction("Train YOLO Model...", self)
        train_action.triggered.connect(self.train_yolo)
        yolo_menu.addAction(train_action)

        # NEW: Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        cross_action = QAction("Cross-Archive Comparison", self)
        cross_action.triggered.connect(self.run_cross_archive_analysis)
        analysis_menu.addAction(cross_action)

    def _connect_signals(self):
        """Connect signals and slots"""
        # Canvas signals
        self.canvas.box_added.connect(self.on_box_added)
        self.canvas.calibration_completed.connect(self.on_calibration_completed)
        self.canvas.roi_selected.connect(self.on_roi_selected)  # NEW
        self.canvas.detection_deleted.connect(self.on_detection_deleted)  # NEW
        self.canvas.box_edited.connect(self._handle_box_edited)
        self.canvas.box_deleted.connect(self._handle_box_deleted)
        self.canvas.undo_requested.connect(self.undo)

        # Control panel signals
        self.control_panel.process_clicked.connect(self.process_annotations)
        self.control_panel.export_coco_clicked.connect(self.export_coco)
        self.control_panel.export_csv_clicked.connect(self.export_csv)
        self.control_panel.undo_clicked.connect(self.undo)
        self.control_panel.redo_clicked.connect(self.redo)
        self.control_panel.clear_clicked.connect(self.clear_all)
        self.control_panel.scale_changed.connect(self.on_scale_changed)
        self.control_panel.calibration_clicked.connect(self.start_calibration)

        # NEW: Connect class_changed signal from control_panel
        self.control_panel.class_changed.connect(self.on_class_changed)  # New method below
        self.control_panel.classes_updated.connect(self.on_classes_updated)

        # Connect tile size change signal
        self.control_panel.tile_size_changed.connect(self.on_tile_size_changed)

        # Debug controls
        self.control_panel.debug_mode_check.toggled.connect(self.toggle_debug_mode)
        self.control_panel.show_debug_btn.clicked.connect(self.show_debug_viewer)

        # Display options
        self.control_panel.show_boxes_check.toggled.connect(
            lambda checked: setattr(self.canvas, 'show_boxes', checked) or self.canvas.update()
        )
        self.control_panel.show_segmentations_check.toggled.connect(
            lambda checked: (
                setattr(self.canvas, 'show_segmentations', checked),
                self.canvas._clear_cache(),
                self.canvas.update()
            )[-1]
        )
        self.control_panel.select_roi_clicked.connect(lambda: self.canvas.set_roi_mode(True))  # NEW
        self.control_panel.run_yolo_clicked.connect(self.run_yolo_inference)  # NEW

        # Tiling controls
        self.control_panel.show_grid_check.toggled.connect(
            lambda checked: setattr(self.canvas, 'show_grid', checked) or self.canvas.update()
        )

        # === NEW: Wire image adjustments (session-only) ===
        self.control_panel.brightness_changed.connect(self.canvas.set_brightness)
        self.control_panel.contrast_changed.connect(self.canvas.set_contrast)
        self.control_panel.saturation_changed.connect(self.canvas.set_saturation)
        self.control_panel.reset_image_adjustments.connect(self.canvas.reset_image_adjustments)

        # NEW: Edit mode and reprocess
        self.control_panel.edit_mode_toggled.connect(self._set_edit_mode)
        self.control_panel.reprocess_clicked.connect(self._reprocess_edited)

    @pyqtSlot()
    def open_image(self):
        """Open image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str):
        """Load image from file"""
        try:
            self.current_image_path = file_path
            self.image_loader = MemoryEfficientImageLoader(file_path)

            # Load full image for both display and processing
            self.current_image = self.image_loader.load_full_image()

            # Update canvas with full resolution image
            self.canvas.set_image(self.current_image)

            # Clear previous annotations and calibration
            self.clear_all()
            self.calibration_line = None
            self.canvas.set_calibration_line(None)
            self.control_panel.update_calibration_status(None)

            # Added: Find or create archive and copy image
            image_name = Path(file_path).stem
            self.current_archive_path = self.archive_manager.find_existing_for_image(image_name)
            if not self.current_archive_path:
                self.current_archive_path = self.archive_manager.create_archive_for_image(file_path)

            # Update current_image_path to the absolute copied version
            copied_name = Path(file_path).name
            archive_dir = os.path.join(self.config['archive']['root_dir'], self.current_archive_path)
            self.current_image_path = os.path.abspath(os.path.join(archive_dir, copied_name))

            logger.info(f"Loaded image: {file_path}")
            logger.info(f"Image size: {self.current_image.shape}")

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    @pyqtSlot()
    def start_calibration(self):
        """Start calibration mode"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        self.canvas.set_calibration_mode(True)

    @pyqtSlot(object)
    def on_calibration_completed(self, calibration_line):
        """Handle calibration completion"""
        self.calibration_line = calibration_line
        self.control_panel.update_calibration_status(calibration_line)

        # CRITICAL FIX: Update the scale input field with calibration value
        self.control_panel.scale_input.setText(str(calibration_line.pixels_per_unit))

        self.annotation_manager.measurement_engine.update_scale(calibration_line.pixels_per_unit)

        # Update canvas
        self.canvas.set_calibration_line(calibration_line)

    # In main_window.py, update the on_box_added method:

    @pyqtSlot(BoundingBox)
    def on_box_added(self, box: BoundingBox):
        """Handle box added from canvas"""
        # Get the currently selected class from control panel
        current_class_id = self.control_panel.class_combo.currentIndex()

        # Add box through annotation manager with the selected class
        new_box = self.annotation_manager.add_box(
            box.x1, box.y1, box.x2, box.y2,
            class_id=current_class_id  # Pass the selected class
        )

        # Update canvas
        self.canvas.add_box(new_box)

        # Update statistics
        self.update_statistics()

    @pyqtSlot()
    def process_annotations(self):
        """Process annotations with SAM"""
        if not self.annotation_manager.boxes:
            QMessageBox.warning(self, "Warning", "No boxes to process")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return

        # Get scale - will warn if using default
        scale = self.control_panel.get_scale()
        if scale is None:
            QMessageBox.critical(self, "Error", "No scale available. Please calibrate or set scale manually.")
            return

        # UPDATE TILE SIZE FROM COMBO BOX
        self.config['tiling']['base_tile_size'] = self.control_panel.get_tile_size()
        self.annotation_manager.tile_manager.base_tile_size = self.control_panel.get_tile_size()

        # Debug print
        logger.info(f"Processing with tile size: {self.control_panel.get_tile_size()}")
        logger.info(f"Using scale: {scale:.2f} px/cm")
        logger.info(f"Surgical approach: Each box gets best mask")

        # Create progress dialog
        progress = QProgressDialog("Processing annotations...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        # Create processing thread
        self.processing_thread = ProcessingThread(
            self.annotation_manager,
            self.current_image,
            scale
        )

        # Connect signals
        self.processing_thread.progress.connect(
            lambda v, m, t: progress.setValue(v) or progress.setLabelText(t)
        )
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)

        # Start processing
        self.processing_thread.start()

        # Handle cancellation
        progress.canceled.connect(self.processing_thread.terminate)

    @pyqtSlot(list)
    def on_processing_finished(self, segmentations):
        """Handle processing completion"""
        try:
            # Update canvas
            self.canvas.set_segmentations(segmentations)

            # NEW: Clear cache to ensure class colors show up immediately
            self.canvas._clear_cache()
            self.canvas.update()

            # Update statistics
            self.update_statistics()

            # Save to archive after processing
            if self.current_archive_path:
                project = self.annotation_manager.get_project_data(
                    self.current_image_path,
                    self.current_image.shape if self.current_image is not None else (0, 0, 0),
                    self.control_panel.get_scale()
                )
                project.archive_path = self.current_archive_path
                project.calibration_line = self.calibration_line
                self.archive_manager.save_data_products(project, self.current_archive_path)

            # Show debug viewer if in debug mode and data available
            logger.info(
                f"Debug mode check: {self.config.get('debug', False)} - Will open viewer: {self.config.get('debug', False) and self.annotation_manager.last_processed_tiles}")
            if self.config.get('debug', False) and self.annotation_manager.last_processed_tiles:
                self.show_debug_viewer()

            if segmentations:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Generated {len(segmentations)} segmentations"
                )
            else:
                self._show_no_segmentations_message()

        except Exception as e:
            logger.error(f"Error in on_processing_finished: {e}")
            import traceback
            traceback.print_exc()

    def _show_no_segmentations_message(self):
        """Show no segmentations message"""
        QMessageBox.warning(
            self,
            "No Segmentations Found",
            "No segmentations were generated. Possible causes:\n\n"
            "• SAM generated masks outside outside boxes (spillover; try strict_clip=False)\n"
            "• Weak object boundaries or similar background\n"
            "• Boxes too small—try larger/padded boxes\n"
            "• Try different tile sizes or check logs\n\n"
            "Using surgical approach: best mask per box (with point fallback and unclip fallback)"
        )

    @pyqtSlot(str)
    def on_processing_error(self, error_msg):
        """Handle processing error"""
        QMessageBox.critical(self, "Processing Error", f"Error: {error_msg}")

    @pyqtSlot()
    def export_coco(self):
        """Export annotations in COCO format"""
        if not self.annotation_manager.segmentations and not self.annotation_manager.all_boxes:
            return

        # Compute default path using archive directory if available
        default_path = "annotations.json"  # Fallback
        if self.current_archive_path:
            archive_dir = os.path.join(self.config['archive']['root_dir'], self.current_archive_path)
            if os.path.exists(archive_dir):
                default_path = os.path.join(archive_dir, "annotations.json")
            else:
                logger.warning(f"Archive directory not found: {archive_dir}. Falling back to default path.")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export COCO",
            default_path,
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                scale = self.control_panel.get_scale()
                project = self.annotation_manager.get_project_data(
                    self.current_image_path,
                    self.current_image.shape,
                    scale
                )
                self.export_manager.export_coco(project, file_path)
                QMessageBox.information(self, "Success", "COCO export completed")
            except Exception as e:
                logger.error(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", str(e))

    @pyqtSlot()
    def export_csv(self):
        """Export measurements to CSV"""
        if not self.annotation_manager.segmentations and not self.annotation_manager.all_boxes:
            return

        # Compute default path using archive directory if available
        default_path = "measurements.csv"  # Fallback
        if self.current_archive_path:
            archive_dir = os.path.join(self.config['archive']['root_dir'], self.current_archive_path)
            if os.path.exists(archive_dir):
                default_path = os.path.join(archive_dir, "measurements.csv")
            else:
                logger.warning(f"Archive directory not found: {archive_dir}. Falling back to default path.")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            default_path,
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                scale = self.control_panel.get_scale()
                project = self.annotation_manager.get_project_data(
                    self.current_image_path,
                    self.current_image.shape,
                    scale
                )
                self.export_manager.export_csv(project, file_path)

                # NEW: Generate visualization report if checkbox is checked
                if self.control_panel.get_generate_report():
                    try:
                        import pandas as pd
                        from utils.visualization_utils import generate_report
                        df = pd.read_csv(file_path)
                        save_dir = os.path.dirname(file_path)
                        generate_report(df, save_dir)
                        logger.info(f"Visualization report generated in {save_dir}")
                    except Exception as e:
                        logger.error(f"Failed to generate visualization report: {e}")
                        QMessageBox.warning(self, "Report Warning", f"CSV exported successfully, but report generation failed: {str(e)}")

                QMessageBox.information(self, "Success", "CSV export completed")
            except Exception as e:
                logger.error(f"Export error: {e}")
                QMessageBox.critical(self, "Export Error", str(e))

    @pyqtSlot()
    def save_project(self):
        """Save current project"""
        if self.current_image_path is None or self.current_archive_path is None:
            QMessageBox.warning(self, "Warning", "No project to save")
            return

        try:
            project = self.annotation_manager.get_project_data(
                self.current_image_path,
                self.current_image.shape,
                self.control_panel.get_scale()
            )
            project.calibration_line = self.calibration_line
            project.archive_path = self.current_archive_path
            project.metadata['classes'] = self.classes  # NEW: Set classes in metadata
            self.archive_manager.save_data_products(project, self.current_archive_path)
            self.annotation_manager.is_dirty = False
            QMessageBox.information(self, "Success", "Project saved to archive")
        except Exception as e:
            logger.error(f"Save error: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    @pyqtSlot()
    def auto_save(self):
        """Auto-save to archive"""
        if self.current_archive_path and self.annotation_manager.is_dirty:
            try:
                project = self.annotation_manager.get_project_data(
                    self.current_image_path,
                    self.current_image.shape if self.current_image is not None else (0, 0, 0),
                    self.control_panel.get_scale()
                )
                project.calibration_line = self.calibration_line
                project.archive_path = self.current_archive_path
                project.metadata['classes'] = self.classes  # NEW: Set classes in metadata
                self.archive_manager.save_data_products(project, self.current_archive_path, is_auto=True)
                self.annotation_manager.is_dirty = False
                logger.info("Auto-saved to archive")
            except Exception as e:
                logger.error(f"Auto-save error: {e}")

    @pyqtSlot()
    def load_project(self):
        """Load project from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                project = self.project_manager.load_project(file_path)
                if project:
                    # If image_path is relative, resolve it assuming it's in the same dir as the JSON
                    json_dir = Path(file_path).parent
                    absolute_image_path = str(json_dir / Path(project.image_path).name)
                    self.load_image(absolute_image_path)

                    # NEW: Restore classes from metadata
                    self.classes = project.metadata.get('classes', [])

                    # Restore annotations with IDs
                    self.annotation_manager.restore_from_project(project)  # Added call to new method

                    # Make sure canvas has reference to annotation manager
                    self.canvas.set_annotation_manager(self.annotation_manager)

                    # Update UI
                    for box in project.all_boxes:
                        self.canvas.add_box(box)
                    self.canvas.set_segmentations(project.segmentations)

                    # Restore calibration if present
                    if project.calibration_line:
                        self.calibration_line = project.calibration_line
                        self.canvas.set_calibration_line(project.calibration_line)
                        self.control_panel.update_calibration_status(project.calibration_line)
                        self.annotation_manager.measurement_engine.update_scale(
                            project.calibration_line.pixels_per_unit
                        )
                    elif project.scale_px_per_cm:
                        self.control_panel.scale_input.setText(str(project.scale_px_per_cm))

                    self.update_statistics()

                    QMessageBox.information(self, "Success", "Project loaded")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to load project")
            except Exception as e:
                logger.error(f"Load error: {e}")
                QMessageBox.critical(self, "Load Error", str(e))

    @pyqtSlot()
    def browse_archives(self):
        """Browse and load from archive"""
        archives = self.archive_manager.list_archives()
        if not archives:
            QMessageBox.warning(self, "No Archives", "No archive folders found")
            return

        # Simple dialog for selection
        selected, ok = QInputDialog.getItem(self, "Select Archive", "Archives:", archives, 0, False)
        if ok and selected:
            self.load_archive(selected)

    def load_archive(self, subfolder: str):
        """Load from archive subfolder"""
        latest_json = self.archive_manager.get_latest_json(subfolder)
        if not latest_json:
            QMessageBox.warning(self, "No Data", "No JSON files in archive")
            return

        project = self.project_manager.load_project(latest_json)
        if project:
            # Resolve relative image_path to absolute for loading
            absolute_image_path = self.archive_manager.get_absolute_image_path(subfolder, project.image_path)
            self.load_image(absolute_image_path)  # This will load the copied image
            self.current_archive_path = subfolder

            # NEW: Restore classes from metadata
            self.classes = project.metadata.get('classes', [])

            self.annotation_manager.restore_from_project(project)  # Added
            for box in project.all_boxes:
                self.canvas.add_box(box)
            self.canvas.set_segmentations(project.segmentations)
            if project.calibration_line:
                self.calibration_line = project.calibration_line
                self.canvas.set_calibration_line(project.calibration_line)
                self.control_panel.update_calibration_status(project.calibration_line)
                self.annotation_manager.measurement_engine.update_scale(project.calibration_line.pixels_per_unit)
            self.update_statistics()
            QMessageBox.information(self, "Success", f"Loaded from archive: {subfolder}")
        else:
            QMessageBox.warning(self, "Warning", "Failed to load from archive")

    @pyqtSlot()
    def undo(self):
        """Undo last action"""
        self.annotation_manager.undo()
        self.refresh_display()

    @pyqtSlot()
    def redo(self):
        """Redo action"""
        self.annotation_manager.redo()
        self.refresh_display()

    @pyqtSlot()
    def clear_all(self):
        """Clear all annotations"""
        self.annotation_manager.clear_all()
        self.canvas.clear_all()
        # Note: We do NOT clear calibration here - it persists across annotation clears
        self.update_statistics()

    @pyqtSlot(float)
    def on_scale_changed(self, scale: float):
        """Handle scale change"""
        self.annotation_manager.measurement_engine.update_scale(scale)

    def refresh_display(self):
        """Refresh canvas display"""
        self.canvas.clear_all()
        for box in self.annotation_manager.all_boxes:  # Updated to all_boxes
            self.canvas.add_box(box)
        self.canvas.set_segmentations(self.annotation_manager.segmentations)
        self.update_statistics()

    def update_statistics(self):
        """Update statistics display"""
        num_boxes = len(self.annotation_manager.all_boxes)  # Updated
        num_segmentations = len(self.annotation_manager.segmentations)
        self.control_panel.update_statistics(num_boxes, num_segmentations)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About EchAIno Annotator",
            "EchAIno Annotator v1.0\n\n"
            "An annotation tool for small marine organisms using\n"
            "Segment Anything Model (SAM) with box prompts\n\n"
            "Surgical approach: Best mask per box\n"
            "No parameter tuning required\n\n"
            "Created with PyQt6 and SAM\n"
            "With YOLOv11 integration"
        )

    def show_debug_viewer(self):
        """Show the debug tile viewer"""
        try:
            if self.debug_viewer is None:
                self.debug_viewer = DebugTileViewer(self)
                self.debug_viewer.closed.connect(self._on_debug_viewer_closed)

                # Pass initial config (no parameters needed for surgical approach)
                self.debug_viewer.set_initial_parameters(self.config)

            # Check if annotation manager has debug data
            if not self.annotation_manager.last_processed_tiles:
                QMessageBox.warning(self, "No Debug Data", "Run processing first to generate debug data.")
                return

            # Update with latest data
            self.debug_viewer.set_tile_data(
                self.annotation_manager.last_processed_tiles,
                self.annotation_manager.debug_tile_images,
                self.annotation_manager.debug_tile_results
            )

            self.debug_viewer.show()
            self.debug_viewer.raise_()
            self.debug_viewer.activateWindow()

        except Exception as e:
            logger.error(f"Error showing debug viewer: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self,
                "Debug Viewer Error",
                f"Could not open debug viewer:\n{str(e)}"
            )

    def toggle_debug_mode(self, checked):
        """Toggle debug mode on/off"""
        self.config['debug'] = checked
        self.annotation_manager.debug_mode = checked

        if checked:
            QMessageBox.information(
                self,
                "Debug Mode Enabled",
                "Debug mode is now ON.\n\n"
                "The tile debug viewer will automatically open after processing.\n\n"
                "Additional debug files will be saved to the 'debug_output' directory."
            )

    def _on_debug_viewer_closed(self):
        """Handle debug viewer close"""
        self.debug_viewer = None

    @pyqtSlot(int)
    def on_tile_size_changed(self, size: int):
        """Handle tile size change from combo box"""
        logger.info(f"Tile size changed to: {size}")
        self.canvas.tile_size = size
        self.annotation_manager.tile_manager.base_tile_size = size
        self.canvas.update()

    # NEW: Train method
    @pyqtSlot()
    def train_yolo(self):
        dialog = YoloTrainDialog(self.config, self.archive_manager.list_archives())
        if dialog.exec():
            selected_archives = dialog.get_selected_archives()
            params = dialog.get_params()
            params['classes'] = self.classes  # NEW: Pass classes to params for trainer

            # Create and assign to self for strong reference
            self.yolo_train_thread = QThread()
            self.yolo_train_worker = YoloTrainWorker(self.yolo_trainer, selected_archives, params)

            # Move worker to thread
            self.yolo_train_worker.moveToThread(self.yolo_train_thread)

            # Connect signals
            self.yolo_train_thread.started.connect(self.yolo_train_worker.run)
            self.yolo_train_worker.finished.connect(self.yolo_train_thread.quit)
            self.yolo_train_worker.finished.connect(self.on_yolo_train_finished)
            self.yolo_train_worker.error.connect(self.on_yolo_train_error)  # New: Handle errors
            self.yolo_train_worker.finished.connect(self.yolo_train_worker.deleteLater)  # Safe cleanup
            self.yolo_train_thread.finished.connect(self.yolo_train_thread.deleteLater)  # Safe cleanup

            # Start thread
            logger.info("Starting YOLO training thread")
            self.yolo_train_thread.start()

    def on_yolo_train_finished(self, model_path):
        logger.info("YOLO training completed successfully")
        QMessageBox.information(self, "Training Complete", f"Model saved to {model_path}")
        # Cleanup references
        self.yolo_train_worker = None
        self.yolo_train_thread = None

    def on_yolo_train_error(self, err_msg):  # New: Error handler
        logger.error(f"YOLO training failed: {err_msg}")
        QMessageBox.critical(self, "Training Error", err_msg)
        # Cleanup on error too
        if self.yolo_train_thread.isRunning():
            self.yolo_train_thread.quit()
            self.yolo_train_thread.wait()  # Wait for clean exit
        self.yolo_train_worker = None
        self.yolo_train_thread = None

    # NEW: Inference connection
    @pyqtSlot()
    def run_yolo_inference(self):
        model_path = self.control_panel.get_model_path()
        if not model_path or not self.current_roi:
            QMessageBox.warning(self, "Error", "Select model and ROI first")
            return

        # Log batch size from config for debugging
        batch_size = self.config.get('yolo', {}).get('sahi_batch_size', 1)
        logger.info(f"Running YOLO inference with SAHI batch size: {batch_size}")

        # New: Busy dialog for UI feedback
        self.inference_progress = QProgressDialog("Running SAHI inference...", None, 0, 0, self)
        self.inference_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.inference_progress.setMinimumDuration(0)
        self.inference_progress.setRange(0, 0)  # Set to busy mode (spinner, no progress bar)
        self.inference_progress.show()

        # Create and assign to self for strong reference
        self.yolo_inference_thread = QThread()
        self.yolo_inference_worker = YoloInferenceWorker(self.yolo_inferencer, self.current_image, self.current_roi,
                                                         model_path)

        # Move worker to thread
        self.yolo_inference_worker.moveToThread(self.yolo_inference_thread)

        # Connect signals (existing + new for dialog close)
        self.yolo_inference_thread.started.connect(self.yolo_inference_worker.run)
        self.yolo_inference_worker.detections_ready.connect(self.on_yolo_detections_ready)
        self.yolo_inference_worker.error.connect(self.on_yolo_inference_error)
        self.yolo_inference_worker.finished.connect(self.inference_progress.close)  # New: Close on finish
        self.yolo_inference_worker.error.connect(lambda _: self.inference_progress.close())  # New: Close on error

        # Add these for safe cleanup (existing)
        self.yolo_inference_worker.finished.connect(self.yolo_inference_thread.quit)
        self.yolo_inference_worker.finished.connect(lambda: self.yolo_inference_thread.wait(5000))
        self.yolo_inference_worker.finished.connect(self.yolo_inference_worker.deleteLater)
        self.yolo_inference_thread.finished.connect(self.yolo_inference_thread.deleteLater)

        # Start thread
        logger.info("Starting YOLO inference thread")
        self.yolo_inference_thread.start()

    def on_yolo_detections_ready(self, detections):
        logger.info("YOLO inference completed successfully")
        self.annotation_manager.add_yolo_detections(detections)

        # NEW: Get new segmentations (last len(detections) items)
        num_new = len(detections)
        new_segs = self.annotation_manager.segmentations[-num_new:]

        # NEW: Apply measurements with current scale (if set)
        scale = self.control_panel.get_scale()
        if scale:
            self.annotation_manager.measurement_engine.update_scale(scale)
            self.annotation_manager.measurement_engine.batch_calculate(new_segs)
            logger.info(f"Computed scaled measurements for {len(new_segs)} YOLO detections")
        else:
            logger.warning("No scale set; YOLO measurements will be pixel-based only (scaling deferred)")

        # NEW: Add YOLO boxes to canvas for visibility and editability
        new_boxes = self.annotation_manager.all_boxes[-num_new:]  # Last added boxes (one per detection)
        for box in new_boxes:
            self.canvas.add_box(box)  # Now boxes will be in canvas.boxes for drawing/editing

        self.canvas.set_segmentations(self.annotation_manager.segmentations)

        # NEW: Clear cache to ensure YOLO class colors show up
        self.canvas._clear_cache()
        self.canvas.update()

        self.update_statistics()

        # Added: Auto-save after YOLO/SAHI inference
        self.auto_save()

    def on_yolo_inference_error(self, err_msg):  # New: Error handler
        logger.error(f"YOLO inference failed: {err_msg}")
        QMessageBox.critical(self, "Inference Error", err_msg)
        # Cleanup on error too
        if self.yolo_inference_thread.isRunning():
            self.yolo_inference_thread.quit()
        self.yolo_inference_worker = None
        self.yolo_inference_thread = None

    # NEW: Handlers
    def on_roi_selected(self, roi: BoundingBox):
        self.current_roi = roi
        # Optional: Draw the final ROI on canvas or log it
        logger.info(f"ROI selected: {roi.x1}, {roi.y1} to {roi.x2}, {roi.y2}")
        self.canvas.update()  # Refresh to show final box if needed

    def on_detection_deleted(self, seg_id: int):
        self.annotation_manager.segmentations = [s for s in self.annotation_manager.segmentations if s.id != seg_id]
        self.canvas.set_segmentations(self.annotation_manager.segmentations)
        self.update()

    # NEW: Handler for class change from control_panel

    def on_classes_updated(self, new_classes: list):
        """Persist class list, update managers, and refresh UI."""
        try:
            if not isinstance(new_classes, list):
                return
            self.config['classes'] = list(new_classes)
            try:
                self._save_config()
            except Exception:
                pass
            # Update annotation manager
            if hasattr(self, 'annotation_manager') and self.annotation_manager:
                if hasattr(self.annotation_manager, 'update_classes'):
                    self.annotation_manager.update_classes(new_classes)
                else:
                    self.annotation_manager.classes = list(new_classes)
            # Ensure UI combo is synced
            try:
                self.control_panel.class_combo.blockSignals(True)
                self.control_panel.class_combo.clear()
                self.control_panel.class_combo.addItems(new_classes or ['Class 0'])
            finally:
                self.control_panel.class_combo.blockSignals(False)
        except Exception as e:
            try:
                logger.error(f'on_classes_updated failed: {e}')
            except Exception:
                pass

    def on_class_changed(self, class_id: int):
        """Handle class selection change (e.g., set global current class for new annotations)"""
        logger.info(f"Class changed to: {class_id}")
        # Optional: Propagate to canvas or manager if needed for live preview

    def _set_edit_mode(self, enabled: bool):
        if self.mode_lock.tryLock():  # Prevent concurrent changes
            self.current_mode = 'edit' if enabled else 'normal'
            self.canvas.set_edit_mode(enabled)
            if enabled:
                try:
                    self.canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)
                except Exception:
                    self.canvas.setFocus()
            self.mode_lock.unlock()

    def _handle_box_edited(self, box_id: int, x1: float, y1: float, x2: float, y2: float):
        command = EditBoxCommand(self.annotation_manager, box_id, x1, y1, x2, y2)
        self.annotation_manager._execute_command(command)
        self.refresh_display()

    def _handle_box_deleted(self, box_id: int):
        command = DeleteBoxCommand(self.annotation_manager, box_id)
        self.annotation_manager._execute_command(command)
        self.refresh_display()

    def _reprocess_edited(self):
        # Call process_annotations (perhaps filter to edited boxes via status flag)
        self.process_annotations()  # Or customized call

    @pyqtSlot()
    def run_cross_archive_analysis(self):
        dialog = CrossArchiveAnalysisDialog(self.config, self.archive_manager.list_archives(), self)
        if dialog.exec():
            selected_archives = dialog.get_selected_archives()
            params = dialog.get_params()

            if not selected_archives:
                QMessageBox.warning(self, "No Selection", "Please select at least one archive.")
                return

            # Setup progress dialog
            self.analysis_progress = QProgressDialog("Running cross-archive analysis...", "Cancel", 0, len(selected_archives), self)
            self.analysis_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.analysis_progress.setMinimumDuration(0)

            # Create thread and worker (new class below)
            self.analysis_thread = QThread()
            self.analysis_worker = CrossArchiveAnalysisWorker(
                self.archive_manager, self.project_manager, self.export_manager,
                selected_archives, params, self.config
            )
            self.analysis_worker.moveToThread(self.analysis_thread)

            self.analysis_thread.started.connect(self.analysis_worker.run)
            self.analysis_worker.progress.connect(self.analysis_progress.setValue)
            self.analysis_worker.finished.connect(self.on_analysis_finished)
            self.analysis_worker.error.connect(self.on_analysis_error)
            self.analysis_worker.finished.connect(self.analysis_thread.quit)
            self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
            self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)

            self.analysis_progress.canceled.connect(self.analysis_thread.quit)
            self.analysis_thread.start()

    def on_analysis_finished(self, report_path):
        self.analysis_progress.close()
        QMessageBox.information(self, "Analysis Complete", f"Report saved to {report_path}")
        # Optional: Open the report folder or HTML

    def on_analysis_error(self, err_msg):
        self.analysis_progress.close()
        QMessageBox.critical(self, "Analysis Error", err_msg)
        if self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()

    def keyPressEvent(self, event: QKeyEvent):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and event.key() == Qt.Key.Key_E:
            # Toggle edit mode
            new_mode = not self.control_panel.edit_mode_check.isChecked()  # Get current from checkbox
            self.control_panel.edit_mode_check.setChecked(new_mode)  # Sync checkbox
            self._set_edit_mode(new_mode)  # Call existing method to propagate to canvas
            # Optional: Status bar feedback
            self.statusBar().showMessage(f"Edit Mode: {'ON' if new_mode else 'OFF'}", 2000)
        super().keyPressEvent(event)
