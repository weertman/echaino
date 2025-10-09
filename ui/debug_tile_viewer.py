from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QPushButton, QGroupBox, QTextEdit,
                             QCheckBox, QDoubleSpinBox, QSpinBox,
                             QFormLayout, QTabWidget, QWidget, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QFont, QColor, QMouseEvent
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Set
from data.models import Tile, BoundingBox, Segmentation
import logging

logger = logging.getLogger(__name__)


class InteractiveTileDisplay(QLabel):
    """Custom QLabel that handles mouse events for mask interaction"""

    mask_hovered = pyqtSignal(int, int)  # tile_id, mask_index
    mask_clicked = pyqtSignal(int, int)  # tile_id, mask_index

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.current_pixmap = None
        self.display_scale = 1.0
        self.mask_regions = {}  # Store mask contours for hit detection
        self.current_tile_id = None

    def set_image(self, pixmap: QPixmap, tile_id: int, mask_regions: Dict):
        """Set the displayed image and mask regions for hit detection"""
        self.current_pixmap = pixmap
        self.current_tile_id = tile_id
        self.mask_regions = mask_regions

        # Scale to fit
        scaled_pixmap = pixmap.scaled(600, 600, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.display_scale = scaled_pixmap.width() / pixmap.width()
        self.setPixmap(scaled_pixmap)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for hover detection"""
        if not self.current_pixmap or not self.mask_regions:
            return

        # Convert mouse position to original image coordinates
        pos = event.position()
        img_x = int(pos.x() / self.display_scale)
        img_y = int(pos.y() / self.display_scale)

        # Find which masks contain this point
        hovered_masks = []
        for mask_key, regions in self.mask_regions.items():
            if 'contours' in regions:
                for contour in regions['contours']:
                    # Check if point is inside contour
                    result = cv2.pointPolygonTest(contour, (img_x, img_y), False)
                    if result >= 0:  # Inside or on edge
                        hovered_masks.append(mask_key)
                        break

        # Emit signal for the first hovered mask
        if hovered_masks and self.current_tile_id is not None:
            # mask_key is formatted as "box_{box_id}_mask_{mask_idx}"
            parts = hovered_masks[0].split('_')
            if len(parts) >= 4:
                mask_idx = int(parts[3])
                self.mask_hovered.emit(self.current_tile_id, mask_idx)
        else:
            self.mask_hovered.emit(-1, -1)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse click"""
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.current_pixmap or not self.mask_regions:
                return

            pos = event.position()
            img_x = int(pos.x() / self.display_scale)
            img_y = int(pos.y() / self.display_scale)

            for mask_key, regions in self.mask_regions.items():
                if 'contours' in regions:
                    for contour in regions['contours']:
                        result = cv2.pointPolygonTest(contour, (img_x, img_y), False)
                        if result >= 0:
                            parts = mask_key.split('_')
                            if len(parts) >= 4:
                                mask_idx = int(parts[3])
                                self.mask_clicked.emit(self.current_tile_id, mask_idx)
                                return


class DebugTileViewer(QDialog):
    """Debug window for viewing tile processing results - Simplified for surgical approach"""

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tile Segmentation Debug Viewer - Surgical Approach")
        self.setGeometry(100, 100, 1200, 900)

        # Data storage
        self.tiles: List[Tile] = []
        self.tile_images: Dict[int, np.ndarray] = {}
        self.tile_results: Dict[int, List[Dict]] = {}
        self.current_tile_index = 0
        self.show_raw_sam = False

        # Interactive state
        self.hovered_mask_info = None
        self.selected_mask_info = None
        self.mask_regions = {}

        # UI components
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QHBoxLayout()

        # Left side - interactive viewer
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Tile navigation
        nav_group = QGroupBox("Tile Navigation")
        nav_layout = QVBoxLayout()

        self.tile_slider = QSlider(Qt.Orientation.Horizontal)
        self.tile_slider.setMinimum(0)
        self.tile_slider.setMaximum(0)
        self.tile_slider.valueChanged.connect(self._on_slider_changed)
        nav_layout.addWidget(self.tile_slider)

        self.tile_label = QLabel("No tiles loaded")
        self.tile_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.tile_label)

        nav_group.setLayout(nav_layout)
        left_layout.addWidget(nav_group)

        # View options
        view_group = QGroupBox("View Options")
        view_layout = QHBoxLayout()

        self.show_raw_check = QCheckBox("Show All SAM Outputs")
        self.show_raw_check.setToolTip(
            "Show all 3 masks from SAM vs just the selected one"
        )
        self.show_raw_check.toggled.connect(self._on_raw_toggle)
        view_layout.addWidget(self.show_raw_check)

        self.show_labels_check = QCheckBox("Show Mask Labels")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.toggled.connect(self._update_display)
        view_layout.addWidget(self.show_labels_check)

        view_group.setLayout(view_layout)
        left_layout.addWidget(view_group)

        # Interactive tile display
        self.tile_display = InteractiveTileDisplay()
        self.tile_display.setMinimumSize(600, 600)
        self.tile_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tile_display.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.tile_display.mask_hovered.connect(self._on_mask_hovered)
        self.tile_display.mask_clicked.connect(self._on_mask_clicked)
        left_layout.addWidget(self.tile_display)

        # Info panel with tabs
        info_tabs = QTabWidget()
        info_tabs.setMaximumHeight(250)

        # Tile info tab
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_tabs.addTab(self.info_text, "Tile Info")

        # Mask inspector tab
        self.mask_inspector = QTextEdit()
        self.mask_inspector.setReadOnly(True)
        self.mask_inspector.setStyleSheet("font-family: monospace;")
        info_tabs.addTab(self.mask_inspector, "Mask Inspector")

        left_layout.addWidget(info_tabs)

        # Control buttons
        button_layout = QHBoxLayout()

        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self._prev_tile)
        button_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._next_tile)
        button_layout.addWidget(self.next_btn)

        self.export_btn = QPushButton("Export Current")
        self.export_btn.clicked.connect(self._export_current)
        button_layout.addWidget(self.export_btn)

        left_layout.addLayout(button_layout)

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # Right side - simplified info panel
        right_panel = QWidget()
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout()

        # Approach info
        approach_group = QGroupBox("Surgical Approach")
        approach_layout = QVBoxLayout()

        approach_text = QTextEdit()
        approach_text.setReadOnly(True)
        approach_text.setMaximumHeight(150)
        approach_text.setHtml("""
        <p><b>Simple Selection Logic:</b></p>
        <ul>
        <li>Each box generates 3 candidate masks</li>
        <li>The mask with highest confidence is selected</li>
        <li>No parameter tuning required</li>
        <li>Trust SAM's quality scoring</li>
        </ul>
        <p style='color: gray;'>Click on masks in the viewer to inspect their properties</p>
        """)
        approach_layout.addWidget(approach_text)

        approach_group.setLayout(approach_layout)
        right_layout.addWidget(approach_group)

        # Statistics group
        stats_group = QGroupBox("Overall Statistics")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(300)
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        self.setLayout(main_layout)

    def _on_mask_hovered(self, tile_id: int, mask_idx: int):
        """Handle mask hover event"""
        if tile_id < 0 or mask_idx < 0:
            self.hovered_mask_info = None
            self._update_mask_inspector()
            return

        # Find the mask info
        if tile_id in self.tile_results:
            tile_results = self.tile_results[tile_id]
            for result in tile_results:
                raw_masks = result.get('raw_masks', [])
                if mask_idx < len(raw_masks):
                    self.hovered_mask_info = {
                        'tile_id': tile_id,
                        'box_id': result['box_id'],
                        'mask_idx': mask_idx,
                        'mask_data': raw_masks[mask_idx],
                        'box': result.get('box'),
                        'is_selected': mask_idx == result.get('selected_idx')
                    }
                    self._update_mask_inspector()
                    break

    def _on_mask_clicked(self, tile_id: int, mask_idx: int):
        """Handle mask click event"""
        if tile_id >= 0 and mask_idx >= 0:
            self.selected_mask_info = self.hovered_mask_info
        else:
            self.selected_mask_info = None
        self._update_display()

    def _update_mask_inspector(self):
        """Update the mask inspector panel with current hover/selection info"""
        lines = []

        if self.selected_mask_info:
            lines.append("=== SELECTED MASK ===")
            lines.extend(self._format_mask_info(self.selected_mask_info))
            lines.append("")

        if self.hovered_mask_info and self.hovered_mask_info != self.selected_mask_info:
            lines.append("=== HOVERED MASK ===")
            lines.extend(self._format_mask_info(self.hovered_mask_info))

        if not lines:
            lines.append("Hover over or click on a mask to inspect its properties")

        self.mask_inspector.setText("\n".join(lines))

    def _format_mask_info(self, mask_info: Dict) -> List[str]:
        """Format mask information for display"""
        lines = []
        mask_data = mask_info['mask_data']

        lines.append(f"Box ID: {mask_info['box_id']}")
        lines.append(f"Mask Index: {mask_info['mask_idx']}")
        lines.append(f"Confidence Score: {mask_data['score']:.4f}")
        lines.append(f"Area: {mask_data['area']:,} pixels")

        if mask_info['is_selected']:
            lines.append("\n✓ THIS MASK WAS SELECTED")
            lines.append("  (Highest confidence score)")
        else:
            lines.append("\n✗ Not selected")
            lines.append("  (Lower confidence than selected mask)")

        return lines

    def _create_tile_visualization(self, tile: Tile, tile_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Create visualization with overlays"""
        # Start with a copy of the tile image
        if len(tile_image.shape) == 2:
            vis = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2RGB)
        else:
            vis = tile_image.copy()

        # Clear previous mask regions
        self.mask_regions = {}

        # Get results for this tile
        results = self.tile_results.get(tile.id, [])

        # Define colors for different masks
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
        ]

        if self.show_raw_sam:
            # Show ALL raw SAM outputs
            for result in results:
                box_id = result['box_id']
                raw_masks = result.get('raw_masks', [])
                selected_idx = result.get('selected_idx')

                for mask_idx, raw_mask_data in enumerate(raw_masks):
                    mask = raw_mask_data['mask']
                    score = raw_mask_data['score']
                    area = raw_mask_data['area']
                    color = colors[mask_idx % len(colors)]

                    # Check if this is the selected mask
                    is_selected = (mask_idx == selected_idx)

                    # Check if this is the hovered or clicked mask
                    is_hovered = (self.hovered_mask_info and
                                  self.hovered_mask_info['box_id'] == box_id and
                                  self.hovered_mask_info['mask_idx'] == mask_idx)
                    is_clicked = (self.selected_mask_info and
                                  self.selected_mask_info['box_id'] == box_id and
                                  self.selected_mask_info['mask_idx'] == mask_idx)

                    # Create colored overlay
                    overlay = vis.copy()
                    overlay[mask > 0] = color

                    # Blend with original
                    if is_clicked:
                        alpha = 0.5
                    elif is_hovered:
                        alpha = 0.4
                    elif is_selected:
                        alpha = 0.3
                    else:
                        alpha = 0.15

                    vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

                    # Find and draw contours
                    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

                    # Store contours for hit detection
                    mask_key = f"box_{box_id}_mask_{mask_idx}"
                    self.mask_regions[mask_key] = {
                        'contours': contours,
                        'box_id': box_id,
                        'mask_idx': mask_idx,
                        'color': color,
                        'is_selected': is_selected
                    }

                    # Draw contour with appropriate style
                    thickness = 3 if is_clicked else 2 if is_hovered or is_selected else 1
                    line_color = (255, 255, 255) if is_clicked or is_hovered else color
                    cv2.drawContours(vis, contours, -1, line_color, thickness)

                    # Add labels if enabled
                    if self.show_labels_check.isChecked() and contours:
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # Create label
                            label = f"M{mask_idx}: {score:.2f}"
                            if is_selected:
                                label = f"✓ {label}"

                            # Draw label background
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(vis, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2),
                                          (255, 255, 255) if is_hovered or is_clicked else (0, 0, 0), -1)

                            # Draw label text
                            cv2.putText(vis, label, (cx, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 0) if is_hovered or is_clicked else (255, 255, 255), 1)
        else:
            # Show only selected masks
            for result in results:
                selected_idx = result.get('selected_idx')
                if selected_idx is not None and result.get('raw_masks'):
                    mask_data = result['raw_masks'][selected_idx]
                    mask = mask_data['mask']
                    color = (0, 255, 0)  # Green for selected

                    # Create colored overlay
                    overlay = vis.copy()
                    overlay[mask > 0] = color
                    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

                    # Draw contour
                    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, color, 2)

        # Draw boxes (always visible)
        for box in tile.boxes:
            # Convert to tile coordinates
            tile_x1 = int(max(0, box.x1 - tile.x))
            tile_y1 = int(max(0, box.y1 - tile.y))
            tile_x2 = int(min(tile.width, box.x2 - tile.x))
            tile_y2 = int(min(tile.height, box.y2 - tile.y))

            # Draw box
            cv2.rectangle(vis, (tile_x1, tile_y1), (tile_x2, tile_y2), (255, 255, 255), 2)
            cv2.rectangle(vis, (tile_x1, tile_y1), (tile_x2, tile_y2), (0, 0, 0), 1)

            # Draw box ID
            cv2.putText(vis, f"B{box.id}",
                        (tile_x1 + 5, tile_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.putText(vis, f"B{box.id}",
                        (tile_x1 + 5, tile_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1)

        return vis, self.mask_regions

    def _update_display(self):
        """Update the display for current tile"""
        if not self.tiles or self.current_tile_index >= len(self.tiles):
            return

        tile = self.tiles[self.current_tile_index]

        # Update label
        num_successful = sum(1 for r in self.tile_results.get(tile.id, [])
                             if r.get('selected_idx') is not None)
        status = f"{num_successful}/{len(tile.boxes)} successful"
        mode_text = " (All SAM Outputs)" if self.show_raw_sam else " (Selected Only)"

        self.tile_label.setText(f"Tile {tile.id} of {len(self.tiles)} - "
                                f"Position: ({tile.x}, {tile.y}) - "
                                f"Size: {tile.width}×{tile.height} - "
                                f"{status}{mode_text}")

        # Get tile image
        tile_image = self.tile_images.get(tile.id)
        if tile_image is None:
            self.tile_display.setText("No image data for this tile")
            self._update_info(tile, None)
            return

        # Create visualization with mask regions
        vis_image, mask_regions = self._create_tile_visualization(tile, tile_image)

        # Convert to QPixmap
        h, w = vis_image.shape[:2]
        bytes_per_line = 3 * w

        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
        elif vis_image.shape[2] == 4:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGRA2RGB)
        elif vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        qimage = QImage(vis_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Set image in interactive display
        self.tile_display.set_image(pixmap, tile.id, mask_regions)

        # Update info
        results = self.tile_results.get(tile.id, [])
        self._update_info(tile, results)

        # Update statistics
        self._update_statistics()

        # Update mask inspector
        self._update_mask_inspector()

    def set_initial_parameters(self, config: dict):
        """No parameters needed for surgical approach"""
        pass

    def _on_raw_toggle(self, checked: bool):
        """Handle raw SAM output toggle"""
        self.show_raw_sam = checked
        self._update_display()

    def _update_statistics(self):
        """Update statistics display"""
        if not self.tile_results:
            return

        total_boxes = 0
        total_successful = 0
        score_distribution = []

        # Analyze all tiles
        for tile_results in self.tile_results.values():
            total_boxes += len(tile_results)

            for result in tile_results:
                if result.get('selected_idx') is not None:
                    total_successful += 1
                    raw_masks = result.get('raw_masks', [])
                    if result['selected_idx'] < len(raw_masks):
                        score_distribution.append(raw_masks[result['selected_idx']]['score'])

        # Format statistics
        stats_lines = [
            f"Total boxes: {total_boxes}",
            f"Successfully segmented: {total_successful} ({total_successful / total_boxes * 100:.1f}%)" if total_boxes > 0 else "No boxes",
            f"Failed: {total_boxes - total_successful}",
        ]

        if score_distribution:
            stats_lines.extend([
                "",
                "Selected mask scores:",
                f"  Min: {min(score_distribution):.3f}",
                f"  Max: {max(score_distribution):.3f}",
                f"  Mean: {np.mean(score_distribution):.3f}",
                f"  Std: {np.std(score_distribution):.3f}"
            ])

        self.stats_text.setText("\n".join(stats_lines))

    def _update_info(self, tile: Tile, results: Optional[List[Dict]]):
        """Update information panel"""
        info_lines = [
            f"Tile ID: {tile.id}",
            f"Position: ({tile.x}, {tile.y})",
            f"Size: {tile.width} × {tile.height}",
            f"Boxes in tile: {len(tile.boxes)}",
            ""
        ]

        if results:
            info_lines.append("Box Results:")
            for result in results:
                box_id = result['box_id']
                selected_idx = result.get('selected_idx')
                raw_masks = result.get('raw_masks', [])

                info_lines.append(f"\nBox {box_id}:")
                info_lines.append(f"  SAM produced: {len(raw_masks)} masks")

                if selected_idx is not None and selected_idx < len(raw_masks):
                    selected_mask = raw_masks[selected_idx]
                    info_lines.append(f"  Selected: Mask {selected_idx}")
                    info_lines.append(f"  Score: {selected_mask['score']:.3f}")
                    info_lines.append(f"  Area: {selected_mask['area']} pixels")
                else:
                    info_lines.append(f"  Status: No mask selected")

        self.info_text.setText("\n".join(info_lines))

    def set_tile_data(self, tiles: List[Tile],
                      tile_images: Dict[int, np.ndarray],
                      tile_results: Dict[int, List[Dict]]):
        """Set tile data for display"""
        self.tiles = tiles
        self.tile_images = tile_images
        self.tile_results = tile_results

        # Reset interaction state
        self.hovered_mask_info = None
        self.selected_mask_info = None

        # Update slider
        if tiles:
            self.tile_slider.setMaximum(len(tiles) - 1)
            self.tile_slider.setValue(0)
            self.current_tile_index = 0
            self._update_display()
        else:
            self.tile_label.setText("No tiles to display")

    def _prev_tile(self):
        """Go to previous tile"""
        if self.current_tile_index > 0:
            self.tile_slider.setValue(self.current_tile_index - 1)

    def _next_tile(self):
        """Go to next tile"""
        if self.current_tile_index < len(self.tiles) - 1:
            self.tile_slider.setValue(self.current_tile_index + 1)

    def _export_current(self):
        """Export current tile visualization"""
        if not self.tiles or self.current_tile_index >= len(self.tiles):
            return

        tile = self.tiles[self.current_tile_index]
        tile_image = self.tile_images.get(tile.id)

        if tile_image is None:
            return

        # Create visualization
        vis_image, _ = self._create_tile_visualization(tile, tile_image)

        # Save dialog
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Tile Visualization",
            f"tile_{tile.id}_debug.png",
            "PNG Files (*.png)"
        )

        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    def _on_slider_changed(self, value: int):
        """Handle slider value change"""
        self.current_tile_index = value
        self._update_display()

    def closeEvent(self, event):
        """Handle window close"""
        self.closed.emit()
        super().closeEvent(event)