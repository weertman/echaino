# FILE: ui\image_canvas.py
# PATH: D:\\echaino\\ui\\image_canvas.py

from PyQt6.QtWidgets import QWidget, QMessageBox, QInputDialog, QMenu
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import (QPainter, QPixmap, QImage, QPen, QColor, QBrush, QCursor, QPainterPath,
                         QKeySequence, QWheelEvent, QMouseEvent, QPaintEvent, QPolygonF)  # Added QPolygonF
from data.models import BoundingBox, Segmentation, CalibrationLine
from core.annotation_manager import EditBoxCommand  # Added import for EditBoxCommand
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
import logging
import math
from utils.image_utils import generate_distinct_colors  # NEW: Import for class colors

logger = logging.getLogger(__name__)


class ImageCanvas(QWidget):
    """Image display canvas with drawing and zooming"""

    box_added = pyqtSignal(BoundingBox)
    calibration_completed = pyqtSignal(CalibrationLine)
    roi_selected = pyqtSignal(BoundingBox)  # NEW signal
    detection_deleted = pyqtSignal(int)  # NEW: For delete action
    box_edited = pyqtSignal(int, float, float, float, float)  # box_id, new_x1, y1, x2, y2
    box_deleted = pyqtSignal(int)  # box_id
    undo_requested = pyqtSignal()  # NEW: Signal for undo in edit mode

    def __init__(self, config: dict, annotation_manager=None):  # Added annotation_manager parameter
        super().__init__()
        self.config = config
        self.annotation_manager = annotation_manager  # Store annotation manager reference
        self.image: Optional[np.ndarray] = None
        self.boxes: List[BoundingBox] = []
        self.segmentations: List[Segmentation] = []
        self.calibration_line: Optional[CalibrationLine] = None
        self.calibration_mode = False
        self.temp_calibration_start: Optional[QPointF] = None
        self.temp_calibration_end: Optional[QPointF] = None

        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.panning = False
        self.last_mouse_pos: Optional[QPointF] = None
        self.current_mouse_pos: QPointF = QPointF(0, 0)

        self.drawing_box = False
        self.temp_box_start: Optional[QPointF] = None
        self.temp_box_end: Optional[QPointF] = None

        self.show_boxes = True
        self.show_segmentations = True
        self.show_grid = False
        self.tile_size = config['tiling']['base_tile_size']

        self.box_color = QColor(*config['ui']['box_color'])
        self.box_width = config['ui']['box_width']

        # Tile-based rendering cache
        self._render_cache: Dict[Tuple[int, int], QPixmap] = {}
        self._cache_tile_size = 512  # For rendering tiles
        self._base_image_changed = True
        self._segmentations_changed = True

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.roi_mode = False  # NEW
        self.temp_roi_start = None  # NEW
        self.temp_roi_end = None  # NEW
        self.selected_detection_id = None  # NEW: For editing
        self.dragging = False  # NEW: Flag for drag mode

        self.edit_mode = False  # New
        self.selected_box_id = None  # New
        self.drag_start = None  # New: For drag/resize tracking
        self.drag_mode = None  # New: 'move', 'resize_nw', etc.
        self.setMouseTracking(True)  # New: Enable move events without press

        # ===== SESSION-ONLY view adjustments (non-destructive) =====
        # Ranges: [-100, 100]
        # contrast α = 1 + c/100 ; brightness β = b ; saturation S *= 1 + s/100
        self._brightness = 0
        self._contrast   = 0
        self._saturation = 0

    def set_annotation_manager(self, annotation_manager):
        """Set or update the annotation manager reference"""
        self.annotation_manager = annotation_manager

    def set_image(self, image: np.ndarray):
        """Set the image to display"""
        self.image = image
        self._base_image_changed = True
        self._clear_cache()
        self.fit_to_view()
        self.update()

    def set_calibration_mode(self, enabled: bool):
        """Enable/disable calibration mode"""
        self.calibration_mode = enabled
        self.temp_calibration_start = None
        self.temp_calibration_end = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_calibration_line(self, calibration_line: Optional[CalibrationLine]):
        """Set the calibration line"""
        self.calibration_line = calibration_line
        self.update()

    def _clear_cache(self):
        """Clear rendering cache"""
        self._render_cache.clear()

    # ===== View adjustment API (session-only) =====
    def set_brightness(self, value: int):
        """β in [-100, 100]"""
        try:
            v = int(value)
        except Exception:
            v = 0
        v = max(-100, min(100, v))
        if v == self._brightness:
            return
        self._brightness = v
        self._base_image_changed = True
        self._clear_cache()
        self.update()

    def set_contrast(self, value: int):
        """α = 1 + v/100, v in [-100, 100]"""
        try:
            v = int(value)
        except Exception:
            v = 0
        v = max(-100, min(100, v))
        if v == self._contrast:
            return
        self._contrast = v
        self._base_image_changed = True
        self._clear_cache()
        self.update()

    def set_saturation(self, value: int):
        """S *= 1 + v/100, v in [-100, 100]"""
        try:
            v = int(value)
        except Exception:
            v = 0
        v = max(-100, min(100, v))
        if v == self._saturation:
            return
        self._saturation = v
        self._base_image_changed = True
        self._clear_cache()
        self.update()

    def reset_image_adjustments(self):
        changed = (self._brightness != 0) or (self._contrast != 0) or (self._saturation != 0)
        self._brightness = 0
        self._contrast   = 0
        self._saturation = 0
        if changed:
            self._base_image_changed = True
            self._clear_cache()
            self.update()

    def _apply_image_adjustments(self, tile_image: np.ndarray) -> np.ndarray:
        """Apply session-only brightness/contrast/saturation to the tile."""
        if tile_image is None:
            return tile_image
        if self._brightness == 0 and self._contrast == 0 and self._saturation == 0:
            return tile_image

        img = tile_image

        # Brightness/Contrast (fast path for uint8)
        if self._brightness != 0 or self._contrast != 0:
            alpha = max(0.0, 1.0 + (self._contrast / 100.0))  # 0..2
            beta  = float(self._brightness)                   # -100..+100
            try:
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            except Exception:
                img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Saturation (RGB only)
        if self._saturation != 0 and img.ndim == 3 and img.shape[2] >= 3:
            sat_scale = max(0.0, 1.0 + (self._saturation / 100.0))
            try:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                s = (s.astype(np.float32) * sat_scale).clip(0, 255).astype(np.uint8)
                hsv = cv2.merge([h, s, v])
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            except Exception:
                # If conversion fails for any reason, keep brightness/contrast result
                pass

        return img

    def _get_visible_tiles(self) -> List[Tuple[int, int]]:
        """Get list of visible tile indices"""
        if self.image is None:
            return []

        widget_rect = self.rect()
        img_w, img_h = self.image.shape[1], self.image.shape[0]

        # Visible image area in image coordinates
        vis_x1 = max(0, int(-self.offset.x() / self.scale))
        vis_y1 = max(0, int(-self.offset.y() / self.scale))
        vis_x2 = min(img_w, int(vis_x1 + widget_rect.width() / self.scale + 1))
        vis_y2 = min(img_h, int(vis_y1 + widget_rect.height() / self.scale + 1))

        # Tile indices
        tx1 = vis_x1 // self._cache_tile_size
        ty1 = vis_y1 // self._cache_tile_size
        tx2 = math.ceil(vis_x2 / self._cache_tile_size)
        ty2 = math.ceil(vis_y2 / self._cache_tile_size)

        tiles = []
        for ty in range(ty1, ty2):
            for tx in range(tx1, tx2):
                tiles.append((tx, ty))
        return tiles

    def _render_tile(self, tile_x: int, tile_y: int) -> Optional[QPixmap]:
        """Render a single tile"""
        if self.image is None:
            return None

        cache_key = (tile_x, tile_y)

        if cache_key in self._render_cache and not (self._base_image_changed or self._segmentations_changed):
            return self._render_cache[cache_key]

        # Extract tile from full image
        x1 = tile_x * self._cache_tile_size
        y1 = tile_y * self._cache_tile_size
        x2 = min(x1 + self._cache_tile_size, self.image.shape[1])
        y2 = min(y1 + self._cache_tile_size, self.image.shape[0])

        if x1 >= x2 or y1 >= y2:
            return None

        tile_image = self.image[y1:y2, x1:x2].copy()

        # >>> Apply session-only adjustments before creating the QImage
        tile_image = self._apply_image_adjustments(tile_image)

        # Create QImage from tile
        h, w = tile_image.shape[:2]
        bytes_per_line = 3 * w if len(tile_image.shape) == 3 else w
        format = QImage.Format.Format_RGB888 if len(tile_image.shape) == 3 else QImage.Format.Format_Grayscale8
        qimage = QImage(tile_image.tobytes(), w, h, bytes_per_line, format)
        tile_pixmap = QPixmap.fromImage(qimage)

        # Add segmentation overlays if enabled
        if self.show_segmentations and self.segmentations:
            painter = QPainter(tile_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            for seg in self.segmentations:
                # Check if seg overlaps this tile
                if not (seg.bbox[0] < x2 and seg.bbox[0] + seg.bbox[2] > x1 and
                        seg.bbox[1] < y2 and seg.bbox[1] + seg.bbox[3] > y1):
                    continue

                # Draw polygon directly
                if not seg.polygon:
                    continue

                # Set color based on class_id first, then fall back to source-based coloring
                if (self.annotation_manager and
                        hasattr(self.annotation_manager, 'class_colors') and
                        seg.class_id < len(self.annotation_manager.class_colors)):
                    color_tuple = self.annotation_manager.class_colors[seg.class_id]
                    color = QColor(*color_tuple)
                else:
                    # Fallback to source-based coloring if no class colors available
                    if seg.source == 'yolo':
                        color = QColor(255, 0, 0) if seg.confidence < 0.5 else QColor(0, 255, 0)
                    else:
                        color = QColor(255, 0, 255)  # Default for SAM

                pen = QPen(color, 2)
                painter.setPen(pen)

                # Use QPolygonF for efficiency
                points = [QPointF(px - x1, py - y1) for px, py in seg.polygon]
                if len(points) < 3:
                    continue  # Skip invalid polygons
                poly = QPolygonF(points)
                painter.drawPolygon(poly)

                # Optional: Draw confidence if YOLO
                if seg.source == 'yolo':
                    # Find a point to label, e.g., centroid translated
                    cx, cy = seg.centroid
                    tx = cx - x1
                    ty = cy - y1
                    if 0 <= tx < w and 0 <= ty < h:  # Only if in tile
                        painter.setPen(QPen(QColor(255, 255, 255), 1))
                        painter.drawText(tx + 5, ty - 5, f"{seg.confidence:.2f}")

            painter.end()

        self._render_cache[cache_key] = tile_pixmap
        return tile_pixmap

    def fit_to_view(self):
        """Fit image to widget size"""
        if self.image is None:
            return

        widget_size = self.size()
        image_h, image_w = self.image.shape[:2]

        scale_x = widget_size.width() / image_w
        scale_y = widget_size.height() / image_h
        self.scale = min(scale_x, scale_y) * 0.9

        scaled_w = image_w * self.scale
        scaled_h = image_h * self.scale
        self.offset = QPointF(
            (widget_size.width() - scaled_w) / 2,
            (widget_size.height() - scaled_h) / 2
        )

    def add_box(self, box: BoundingBox):
        """Add a box to display"""
        self.boxes.append(box)
        self.update()

    def remove_box(self, box: BoundingBox):
        """Remove a box from display"""
        if box in self.boxes:
            self.boxes.remove(box)
            self.update()

    def set_segmentations(self, segmentations: List[Segmentation]):
        """Update displayed segmentations"""
        logger.info(f"Canvas: Setting {len(segmentations)} segmentations")

        # Debug: Check if segmentations have valid data
        for i, seg in enumerate(segmentations):
            mask_sum = np.sum(seg.mask) if seg.mask is not None else "None"
            logger.debug(
                f"  Segmentation {i}: box_id={seg.box_id}, area={mask_sum}, has_polygon={len(seg.polygon) > 0}")
        self.segmentations = segmentations
        self._segmentations_changed = True
        self._clear_cache()
        self.update()

    def clear_all(self):
        """Clear all annotations"""
        self.boxes.clear()
        self.segmentations.clear()
        self._segmentations_changed = True
        self._clear_cache()
        self.update()

    def image_to_widget(self, image_point: Tuple[float, float]) -> QPointF:
        """Convert image coordinates to widget coordinates"""
        x, y = image_point
        wx = x * self.scale + self.offset.x()
        wy = y * self.scale + self.offset.y()
        return QPointF(wx, wy)

    def widget_to_image(self, widget_point: QPointF) -> Tuple[float, float]:
        """Convert widget coordinates to image coordinates"""
        x = (widget_point.x() - self.offset.x()) / self.scale
        y = (widget_point.y() - self.offset.y()) / self.scale
        return (x, y)

    # NEW: Setter for ROI mode
    def set_roi_mode(self, enabled: bool):
        self.roi_mode = enabled
        self.temp_roi_start = None  # NEW
        self.temp_roi_end = None  # NEW
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)  # Visual feedback
            self.setMouseTracking(True)  # Enable mouse move tracking
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setMouseTracking(False)
        self.update()

    def set_edit_mode(self, enabled: bool):
        self.edit_mode = enabled
        if enabled:
            try:
                self.setFocus(Qt.FocusReason.ShortcutFocusReason)
            except Exception:
                self.setFocus()
        self.update()

    # ------------------------
    # (All mouse/paint methods unchanged below)
    # ------------------------
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        try:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        except Exception:
            self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            if self.calibration_mode:
                # Handle calibration clicks
                image_pos = self.widget_to_image(event.position())
                if self.image is not None:
                    h, w = self.image.shape[:2]
                    if 0 <= image_pos[0] < w and 0 <= image_pos[1] < h:
                        if self.temp_calibration_start is None:
                            self.temp_calibration_start = QPointF(image_pos[0], image_pos[1])
                            logger.info(f"Calibration start point set: ({image_pos[0]:.1f}, {image_pos[1]:.1f})")
                        else:
                            self.temp_calibration_end = QPointF(image_pos[0], image_pos[1])
                            logger.info(f"Calibration end point set: ({image_pos[0]:.1f}, {image_pos[1]:.1f})")
                            self._complete_calibration()
                        self.update()
            elif self.edit_mode and self.annotation_manager:
                img_pos = self.widget_to_image(event.position())  # Convert to image coords
                candidates = []  # Collect boxes containing the point
                for box in self.annotation_manager.all_boxes:  # Use manager's all_boxes
                    if box.x1 <= img_pos[0] <= box.x2 and box.y1 <= img_pos[1] <= box.y2:
                        candidates.append(box)

                if candidates:
                    # Sort by area ascending (smallest first)
                    candidates.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
                    selected_box = candidates[0]  # Pick smallest
                    self.selected_box_id = selected_box.id
                    self.drag_start = (img_pos[0], img_pos[1])

                    # Check for handle hits (existing + expanded for all 8)
                    handle_size = 10 / self.scale
                    mid_x = (selected_box.x1 + selected_box.x2) / 2
                    mid_y = (selected_box.y1 + selected_box.y2) / 2

                    if abs(img_pos[0] - selected_box.x1) < handle_size and abs(
                            img_pos[1] - selected_box.y1) < handle_size:
                        self.drag_mode = 'resize_nw'
                    elif abs(img_pos[0] - selected_box.x2) < handle_size and abs(
                            img_pos[1] - selected_box.y2) < handle_size:
                        self.drag_mode = 'resize_se'
                    elif abs(img_pos[0] - selected_box.x1) < handle_size and abs(
                            img_pos[1] - selected_box.y2) < handle_size:
                        self.drag_mode = 'resize_sw'
                    elif abs(img_pos[0] - selected_box.x2) < handle_size and abs(
                            img_pos[1] - selected_box.y1) < handle_size:
                        self.drag_mode = 'resize_ne'
                    elif abs(img_pos[0] - mid_x) < handle_size and abs(img_pos[1] - selected_box.y1) < handle_size:
                        self.drag_mode = 'resize_n'
                    elif abs(img_pos[0] - mid_x) < handle_size and abs(img_pos[1] - selected_box.y2) < handle_size:
                        self.drag_mode = 'resize_s'
                    elif abs(img_pos[0] - selected_box.x1) < handle_size and abs(img_pos[1] - mid_y) < handle_size:
                        self.drag_mode = 'resize_w'
                    elif abs(img_pos[0] - selected_box.x2) < handle_size and abs(img_pos[1] - mid_y) < handle_size:
                        self.drag_mode = 'resize_e'
                    else:
                        self.drag_mode = 'move'  # Interior non-handle: move

                    # Set cursor based on mode
                    if self.drag_mode == 'move':
                        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                    elif self.drag_mode in ['resize_nw', 'resize_se']:
                        self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
                    elif self.drag_mode in ['resize_ne', 'resize_sw']:
                        self.setCursor(QCursor(Qt.CursorShape.SizeBDiagCursor))
                    elif self.drag_mode in ['resize_n', 'resize_s']:
                        self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
                    elif self.drag_mode in ['resize_e', 'resize_w']:
                        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))

                    self.update()
                    return

                # If no box selected, deselect
                self.selected_box_id = None
                self.update()
            elif self.roi_mode:
                image_pos = self.widget_to_image(event.position())
                if self.image is not None:
                    h, w = self.image.shape[:2]
                    if 0 <= image_pos[0] < w and 0 <= image_pos[1] < h:
                        self.temp_roi_start = QPointF(image_pos[0], image_pos[1])
                        self.temp_roi_end = QPointF(image_pos[0], image_pos[1])  # Start with zero-size box
                        self.dragging = True  # Enable dragging
                        self.update()
            elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Start panning
                self.panning = True
                self.last_mouse_pos = event.position()
            else:
                # Start drawing box
                image_pos = self.widget_to_image(event.position())
                if self.image is not None:
                    h, w = self.image.shape[:2]
                    if 0 <= image_pos[0] < w and 0 <= image_pos[1] < h:
                        self.drawing_box = True
                        self.temp_box_start = QPointF(image_pos[0], image_pos[1])
                        self.temp_box_end = self.temp_box_start

        elif event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self.panning = True
            self.last_mouse_pos = event.position()

        elif event.button() == Qt.MouseButton.RightButton:  # Delete detection
            image_pos = self.widget_to_image(event.position())
            # Find if click is inside a YOLO segmentation
            for seg in self.segmentations:
                if seg.source == 'yolo' and self.point_in_polygon(image_pos, seg.polygon):
                    self.detection_deleted.emit(seg.id)
                    break

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move"""
        self.current_mouse_pos = event.position()

        if self.panning:
            delta = event.position() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.position()
            self.update()
        elif self.drawing_box and self.temp_box_start is not None:
            # Update box end point
            image_pos = self.widget_to_image(event.position())
            self.temp_box_end = QPointF(image_pos[0], image_pos[1])
            self.update()
        elif self.calibration_mode and self.temp_calibration_start is not None:
            # Update temporary end point for preview
            image_pos = self.widget_to_image(event.position())
            self.temp_calibration_end = QPointF(image_pos[0], image_pos[1])
            self.update()
        elif self.edit_mode and self.selected_box_id is not None and self.drag_start and self.annotation_manager:
            img_pos = self.widget_to_image(event.position())
            box = next((b for b in self.annotation_manager.all_boxes if b.id == self.selected_box_id), None)
            if not box:
                return

            dx = img_pos[0] - self.drag_start[0]
            dy = img_pos[1] - self.drag_start[1]

            if self.drag_mode == 'move':
                box.x1 += dx
                box.y1 += dy
                box.x2 += dx
                box.y2 += dy
            elif self.drag_mode == 'resize_se':
                box.x2 += dx
                box.y2 += dy
            # Add logic for other resize modes (e.g., 'resize_nw': box.x1 += dx, box.y1 += dy)
            elif self.drag_mode == 'resize_nw':
                box.x1 += dx
                box.y1 += dy
            elif self.drag_mode == 'resize_n':
                box.y1 += dy
            elif self.drag_mode == 'resize_s':
                box.y2 += dy
            elif self.drag_mode == 'resize_e':
                box.x2 += dx
            elif self.drag_mode == 'resize_w':
                box.x1 += dx
            elif self.drag_mode == 'resize_ne':
                box.x2 += dx
                box.y1 += dy
            elif self.drag_mode == 'resize_sw':
                box.x1 += dx
                box.y2 += dy

            self.drag_start = img_pos  # Update start for next delta
            self.update()  # Redraw during drag
        elif self.roi_mode and self.dragging:
            # Update temporary ROI end point for preview
            image_pos = self.widget_to_image(event.position())
            self.temp_roi_end = QPointF(image_pos[0], image_pos[1])
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing_box and self.temp_box_start and self.temp_box_end:
                # Complete box drawing
                x1 = min(self.temp_box_start.x(), self.temp_box_end.x())
                y1 = min(self.temp_box_start.y(), self.temp_box_end.y())
                x2 = max(self.temp_box_start.x(), self.temp_box_end.x())
                y2 = max(self.temp_box_start.y(), self.temp_box_end.y())

                # Minimum box size (10x10 pixels)
                if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                    # Box will be created by main window
                    self.box_added.emit(BoundingBox(-1, x1, y1, x2, y2))

                self.drawing_box = False
                self.temp_box_start = None
                self.temp_box_end = None
                self.update()
            elif self.edit_mode and self.selected_box_id and self.annotation_manager:
                box = next((b for b in self.annotation_manager.all_boxes if b.id == self.selected_box_id), None)
                if box:
                    self.box_edited.emit(box.id, box.x1, box.y1, box.x2, box.y2)  # Emit to manager for command
                self.drag_start = None
                self.drag_mode = None
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                self.update()
            elif self.roi_mode and self.dragging:
                image_pos = self.widget_to_image(event.position())
                self.temp_roi_end = QPointF(image_pos[0], image_pos[1])
                self.dragging = False

                # Calculate normalized box (in image coordinates)
                x1 = min(self.temp_roi_start.x(), self.temp_roi_end.x())
                y1 = min(self.temp_roi_start.y(), self.temp_roi_end.y())
                x2 = max(self.temp_roi_start.x(), self.temp_roi_end.x())
                y2 = max(self.temp_roi_start.y(), self.temp_roi_end.y())

                # Ensure valid size
                if x2 - x1 > 0 and y2 - y1 > 0:
                    box = BoundingBox(id=-1, x1=x1, y1=y1, x2=x2, y2=y2)  # ID can be set later
                    self.roi_selected.emit(box)  # Emit signal with box
                else:
                    # Invalid box (e.g., click without drag)
                    return

                # Reset mode
                self.set_roi_mode(False)
                self.update()
            else:
                self.panning = False

        elif event.button() in [Qt.MouseButton.MiddleButton]:
            self.panning = False

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        mouse_pos = event.position()
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.15 if zoom_in else 1.0 / 1.15

        new_scale = self.scale * zoom_factor
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))

        if new_scale != self.scale:
            old_image_pos = self.widget_to_image(mouse_pos)
            self.scale = new_scale
            new_widget_pos = self.image_to_widget(old_image_pos)
            self.offset += mouse_pos - new_widget_pos
            self.update()

    def _complete_calibration(self):
        """Complete the calibration process"""
        if self.temp_calibration_start is None or self.temp_calibration_end is None:
            return

        dx = self.temp_calibration_end.x() - self.temp_calibration_start.x()
        dy = self.temp_calibration_end.y() - self.temp_calibration_start.y()
        pixel_distance = np.sqrt(dx * dx + dy * dy)

        if pixel_distance < 10:
            parent_window = self.window()
            QMessageBox.warning(parent_window, "Invalid Calibration",
                                "The calibration line is too short. Please draw a longer line.")
            return

        parent_window = self.window()
        distance, ok = QInputDialog.getDouble(
            parent_window, "Calibration Distance",
            f"The line is {pixel_distance:.1f} pixels long.\n"
            "Enter the real-world distance (in cm):",
            value=1.0, min=0.01, max=10000.0, decimals=2
        )

        if ok and distance > 0:
            calibration_line = CalibrationLine(
                start_x=self.temp_calibration_start.x(),
                start_y=self.temp_calibration_start.y(),
                end_x=self.temp_calibration_end.x(),
                end_y=self.temp_calibration_end.y(),
                real_distance=distance,
                unit="cm"
            )

            self.calibration_line = calibration_line
            self.calibration_completed.emit(calibration_line)

            self.calibration_mode = False
            self.temp_calibration_start = None
            self.temp_calibration_end = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

            QMessageBox.information(
                parent_window, "Calibration Complete",
                f"Scale set to {calibration_line.pixels_per_unit:.2f} px/cm"
            )

        self.update()

    def _draw_calibration_line(self, painter: QPainter):
        """Draw the calibration line"""
        if self.calibration_line is not None:
            start = self.image_to_widget((self.calibration_line.start_x, self.calibration_line.start_y))
            end = self.image_to_widget((self.calibration_line.end_x, self.calibration_line.end_y))

            pen = QPen(QColor(0, 255, 0), 3)
            painter.setPen(pen)
            painter.drawLine(start, end)

            painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.drawEllipse(start, 6, 6)
            painter.drawEllipse(end, 6, 6)

            mid_point = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawText(mid_point.x() + 10, mid_point.y() - 10,
                             f"{self.calibration_line.real_distance:.2f} {self.calibration_line.unit}")
            painter.drawText(mid_point.x() + 10, mid_point.y() + 10,
                             f"{self.calibration_line.pixels_per_unit:.2f} px/{self.calibration_line.unit}")

        if self.calibration_mode and self.temp_calibration_start is not None:
            start = self.image_to_widget((self.temp_calibration_start.x(), self.temp_calibration_start.y()))

            if self.temp_calibration_end is not None:
                end = self.image_to_widget((self.temp_calibration_end.x(), self.temp_calibration_end.y()))
            else:
                end = self.current_mouse_pos

            pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(start, end)

            painter.setBrush(QBrush(QColor(255, 255, 0)))
            painter.drawEllipse(start, 6, 6)

            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawText(10, 30, "Click to set calibration end point")

    def _draw_boxes(self, painter: QPainter):
        """Draw all bounding boxes"""
        if not self.show_boxes:
            return

        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Draw existing boxes
        boxes_to_draw = self.annotation_manager.all_boxes if self.annotation_manager else self.boxes
        for box in boxes_to_draw:
            # Determine color based on class
            if self.annotation_manager and hasattr(self.annotation_manager, 'class_colors') and box.class_id < len(
                    self.annotation_manager.class_colors):
                color_tuple = self.annotation_manager.class_colors[box.class_id]
            else:
                color_tuple = tuple(self.config['ui'].get('box_color', [255, 0, 0]))  # Fallback
            color = QColor(*color_tuple)

            if box.status == "processed_failed":
                pen = QPen(color, self.box_width, Qt.PenStyle.DashLine)  # Dashed for failures
            else:
                pen = QPen(color, self.box_width)
            if box.id == self.selected_box_id:
                pen.setWidth(3)  # Thicker for selected
                pen.setColor(QColor(255, 0, 0))  # Red highlight
            painter.setPen(pen)
            tl = self.image_to_widget((box.x1, box.y1))
            br = self.image_to_widget((box.x2, box.y2))
            rect = QRectF(tl, br)
            painter.drawRect(rect)

            # Draw box ID
            info_text = f"{box.id}"
            if self.annotation_manager and hasattr(self.annotation_manager,
                                                   'classes') and self.annotation_manager.classes:
                if box.class_id < len(self.annotation_manager.classes):
                    class_name = self.annotation_manager.classes[box.class_id]
                    info_text += f" ({class_name[0:2]})"
                else:
                    info_text += f" (Class {box.class_id})"
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            text_pos = tl + QPointF(5, -5)
            painter.setPen(QPen(color, 1))
            painter.drawText(text_pos, info_text)

            # If selected, draw 8 handles (small ellipses at corners/midpoints)
            if box.id == self.selected_box_id:
                handles = [  # Calculate 8 points in widget coords
                    self.image_to_widget((box.x1, box.y1)),  # NW
                    self.image_to_widget(((box.x1 + box.x2) / 2, box.y1)),  # N
                    self.image_to_widget((box.x2, box.y1)),  # NE
                    self.image_to_widget((box.x2, (box.y1 + box.y2) / 2)),  # E
                    self.image_to_widget((box.x2, box.y2)),  # SE
                    self.image_to_widget(((box.x1 + box.x2) / 2, box.y2)),  # S
                    self.image_to_widget((box.x1, box.y2)),  # SW
                    self.image_to_widget((box.x1, (box.y1 + box.y2) / 2)),  # W
                ]
                for h in handles:
                    painter.drawEllipse(h, 5, 5)  # Small circles

        # Draw temporary box being drawn
        if self.drawing_box and self.temp_box_start and self.temp_box_end:
            pen = QPen(self.box_color, self.box_width, Qt.PenStyle.DashLine)
            painter.setPen(pen)

            tl = self.image_to_widget((min(self.temp_box_start.x(), self.temp_box_end.x()),
                                       min(self.temp_box_start.y(), self.temp_box_end.y())))
            br = self.image_to_widget((max(self.temp_box_start.x(), self.temp_box_end.x()),
                                       max(self.temp_box_start.y(), self.temp_box_end.y())))
            rect = QRectF(tl, br)
            painter.drawRect(rect)

    def _draw_tile_grid(self, painter: QPainter):
        """Draw tile grid overlay"""
        if not self.show_grid or self.image is None:
            return

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        overlap = int(self.tile_size * self.config['tiling']['overlap_ratio'])
        step = max(1, self.tile_size - overlap)

        if self.boxes:
            # Show actual tiles that would be generated
            x_coords = []
            y_coords = []
            for box in self.boxes:
                x_coords.extend([box.x1, box.x2])
                y_coords.extend([box.y1, box.y2])

            min_x = max(0, int(min(x_coords)) - self.tile_size // 2)
            max_x = min(self.image.shape[1], int(max(x_coords)) + self.tile_size // 2)
            min_y = max(0, int(min(y_coords)) - self.tile_size // 2)
            max_y = min(self.image.shape[0], int(max(y_coords)) + self.tile_size // 2)

            colors = [QColor(255, 255, 0, 80), QColor(0, 255, 255, 80)]
            color_idx = 0

            for y in range(min_y, max_y, step):
                for x in range(min_x, max_x, step):
                    # Check if any box intersects this tile
                    tile_box = BoundingBox(-1, x, y, x + self.tile_size, y + self.tile_size)
                    has_box = any(tile_box.intersects(box) for box in self.boxes)

                    if has_box:
                        color = colors[color_idx % 2]
                        color_idx += 1

                        tl = self.image_to_widget((x, y))
                        br = self.image_to_widget((
                            min(x + self.tile_size, self.image.shape[1]),
                            min(y + self.tile_size, self.image.shape[0])
                        ))

                        painter.fillRect(QRectF(tl, br), color)
                        painter.setPen(QPen(color.darker(), 2))
                        painter.drawRect(QRectF(tl, br))
        else:
            # Show reference grid
            pen = QPen(QColor(255, 255, 0, 50), 1, Qt.PenStyle.DotLine)
            painter.setPen(pen)

            for x in range(0, self.image.shape[1], step * 2):
                start = self.image_to_widget((x, 0))
                end = self.image_to_widget((x, self.image.shape[0]))
                painter.drawLine(start, end)

            for y in range(0, self.image.shape[0], step * 2):
                start = self.image_to_widget((0, y))
                end = self.image_to_widget((self.image.shape[1], y))
                painter.drawLine(start, end)

        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawText(10, 30, f"Tile: {self.tile_size}px, Step: {step}px")

    # NEW: Helper for point in polygon (using cv2 or similar)
    def point_in_polygon(self, point: Tuple[float, float], polygon: List[List[float]]) -> bool:
        if not polygon:
            return False
        pts = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, point, False) >= 0

    def keyPressEvent(self, event):
        """
        Robust hotkeys in Edit mode:
          - digits 1..9 set classes 0..8
          - digit 0 sets class 9 if >=10 classes, else last class
          - '[' / ']' cycles class on the selected (or under-cursor) box
          - works with top-row and numpad digits
        """
        handled = False
        if self.edit_mode and self.annotation_manager:
            if self.selected_box_id is not None and event.key() == Qt.Key.Key_Delete:
                self.box_deleted.emit(self.selected_box_id)
                self.selected_box_id = None
                self.update()
                handled = True
            elif (event.key() == Qt.Key.Key_Z and
                  (event.modifiers() & Qt.KeyboardModifier.ControlModifier)):
                self.undo_requested.emit()
                handled = True
            elif event.key() in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
                box = self._ensure_selection_under_cursor_if_missing()
                if box:
                    n = max(1, len(self.annotation_manager.classes))
                    cur = getattr(box, 'class_id', 0) or 0
                    delta = -1 if event.key() == Qt.Key.Key_BracketLeft else 1
                    new_cid = (cur + delta) % n
                    self._apply_class_id(box, new_cid)
                    handled = True
            else:
                txt = event.text() or ""
                no_meta = not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                                                    Qt.KeyboardModifier.AltModifier))
                if txt.isdigit() and no_meta:
                    digit = int(txt)
                    new_cid = self._map_digit_to_class(digit)
                    box = self._ensure_selection_under_cursor_if_missing()
                    if box:
                        self._apply_class_id(box, new_cid)
                        handled = True

        if not handled:
            super().keyPressEvent(event)

    def _map_digit_to_class(self, digit: int) -> int:
        """
        Map keyboard digit to a class index.

        Behavior:
          - '1'..'9' -> 0..8
          - '0'      -> 9
        If the annotation manager has a non-empty classes list, we clamp to its max index.
        If there are no classes configured, we DO NOT force everything to 0; we keep the
        natural mapping above (so 3 -> class_id 2, etc.).
        """
        # Base mapping independent of classes
        idx = 9 if digit == 0 else max(0, digit - 1)

        # If classes exist, clamp to last available index; otherwise leave idx as-is
        try:
            num = len(self.annotation_manager.classes) if (self.annotation_manager and
                                                           hasattr(self.annotation_manager, 'classes')) else 0
        except Exception:
            num = 0

        if num > 0:
            idx = min(idx, num - 1)

        return idx

    def _ensure_selection_under_cursor_if_missing(self):
        selected_box = None
        if self.selected_box_id is not None:
            selected_box = next((b for b in self.annotation_manager.all_boxes if b.id == self.selected_box_id), None)
        if selected_box is None and self.annotation_manager:
            img_pos = self.widget_to_image(self.current_mouse_pos)
            cands = [b for b in self.annotation_manager.all_boxes
                     if b.x1 <= img_pos[0] <= b.x2 and b.y1 <= img_pos[1] <= b.y2]
            if cands:
                cands.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
                selected_box = cands[0]
                self.selected_box_id = selected_box.id
        return selected_box

    def _apply_class_id(self, box, new_class_id: int):
        try:
            command = EditBoxCommand(
                self.annotation_manager,
                box.id,
                box.x1, box.y1, box.x2, box.y2,
                new_class_id=new_class_id
            )
            self.annotation_manager._execute_command(command)
            self.update()
            logger.info(f"Changed box {box.id} to class {new_class_id} via hotkey")
            from PyQt6.QtWidgets import QToolTip
            name = (self.annotation_manager.classes[new_class_id]
                    if hasattr(self.annotation_manager, 'classes') and new_class_id < len(
                self.annotation_manager.classes)
                    else str(new_class_id))
            QToolTip.showText(self.mapToGlobal(self.current_mouse_pos.toPoint()),
                              f"Changed to {name}")
        except Exception as e:
            logger.error(f"Failed to apply class {new_class_id} to box {getattr(box, 'id', '?')}: {e}")

    def contextMenuEvent(self, event):
        if self.edit_mode and self.selected_box_id is not None and self.annotation_manager:
            selected_box = next((b for b in self.annotation_manager.all_boxes if b.id == self.selected_box_id), None)
            if selected_box and hasattr(self.annotation_manager, 'classes'):
                menu = QMenu(self)
                for i, class_name in enumerate(self.annotation_manager.classes):
                    action = menu.addAction(f"Set to {class_name} (ID: {i})")
                    action.triggered.connect(lambda checked, cid=i: self._change_box_class(selected_box, cid))
                menu.exec(event.globalPos())

    def _change_box_class(self, box: BoundingBox, new_class_id: int):
        if self.annotation_manager and hasattr(self.annotation_manager, 'classes') and new_class_id < len(
                self.annotation_manager.classes):
            command = EditBoxCommand(
                self.annotation_manager,
                box.id,
                box.x1, box.y1, box.x2, box.y2,  # Unchanged
                new_class_id=new_class_id
            )
            self.annotation_manager._execute_command(command)
            self.update()  # Redraw
            logger.info(f"Changed box {box.id} to class {new_class_id} via menu")

    def paintEvent(self, event: QPaintEvent):
        """Paint the canvas using tile-based rendering"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)

        if self.image is not None:
            visible_tiles = self._get_visible_tiles()

            for tile_x, tile_y in visible_tiles:
                tile_pixmap = self._render_tile(tile_x, tile_y)
                if tile_pixmap is None:
                    continue

                tile_img_x = tile_x * self._cache_tile_size
                tile_img_y = tile_y * self._cache_tile_size

                tile_widget_pos = self.image_to_widget((tile_img_x, tile_img_y))

                tile_widget_w = tile_pixmap.width() * self.scale
                tile_widget_h = tile_pixmap.height() * self.scale

                target_rect = QRectF(
                    tile_widget_pos.x(),
                    tile_widget_pos.y(),
                    tile_widget_w,
                    tile_widget_h
                )
                source_rect = QRectF(0, 0, tile_pixmap.width(), tile_pixmap.height())

                painter.drawPixmap(target_rect, tile_pixmap, source_rect)

        self._segmentations_changed = False
        self._base_image_changed = False

        # Draw overlays
        try:
            self._draw_calibration_line(painter)
        except Exception as e:
            logger.error(f"Error drawing calibration line: {e}")

        if self.show_grid and self.image is not None:
            self._draw_tile_grid(painter)

        # Draw boxes
        self._draw_boxes(painter)

        # Draw temp ROI if in ROI mode
        if self.roi_mode and self.temp_roi_start is not None:
            # Get end point consistently as QPointF
            if self.temp_roi_end is not None:
                end_point = self.temp_roi_end
            else:
                temp_end_tuple = self.widget_to_image(self.current_mouse_pos)
                end_point = QPointF(temp_end_tuple[0], temp_end_tuple[1])

            # Calculate min/max in image coords
            x1 = min(self.temp_roi_start.x(), end_point.x())
            y1 = min(self.temp_roi_start.y(), end_point.y())
            x2 = max(self.temp_roi_start.x(), end_point.x())
            y2 = max(self.temp_roi_start.y(), end_point.y())

            # Convert corners to widget coords
            tl = self.image_to_widget((x1, y1))
            br = self.image_to_widget((x2, y2))

            # Draw dashed rectangle
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(tl, br))

            # Draw start point indicator
            start_widget = self.image_to_widget((self.temp_roi_start.x(), self.temp_roi_start.y()))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.drawEllipse(start_widget, 6, 6)

            # Instructional text
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawText(10, 30, "Drag to select ROI")
