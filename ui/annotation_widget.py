# FILE: ui\annotation_widget.py
# PATH: D:\urchinScanner\ui\annotation_widget.py

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel
from PyQt6.QtCore import pyqtSignal
from data.models import BoundingBox
from typing import List
import logging

logger = logging.getLogger(__name__)


class AnnotationWidget(QWidget):
    """Widget for managing annotations, including point/segmentation overlay and class selection"""

    box_added = pyqtSignal(BoundingBox)  # Signal to emit when box added (with class_id)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.classes = config.get('classes', [])  # Get classes from config
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI for annotation controls"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Label for overlay/info
        layout.addWidget(QLabel("Point/Segmentation Overlay Controls"))

        # NEW: Class selection combo box
        class_layout = QVBoxLayout()
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes or ['Class 0'])  # Fallback if no classes
        class_layout.addWidget(QLabel("Select Class:"))
        class_layout.addWidget(self.class_combo)
        layout.addLayout(class_layout)

        # Button to add annotation (example: add box)
        add_btn = QPushButton("Add Box Annotation")
        add_btn.clicked.connect(self.add_annotation)
        layout.addWidget(add_btn)

        # Additional controls (e.g., for points/segments) can be added here
        # Example: Point addition button/logic

        layout.addStretch()

    def add_annotation(self):
        """Add a new annotation (e.g., box) with selected class"""
        # Assuming this triggers box addition; actual coords would come from canvas or input
        # For demo: Use dummy coords; in real, connect to canvas or input
        class_id = self.class_combo.currentIndex()  # Get selected class_id
        dummy_box = BoundingBox(id=-1, x1=0, y1=0, x2=100, y2=100)  # Dummy; replace with real
        # Emit signal with box (class_id set in manager, but can pre-set here if needed)
        self.box_added.emit(dummy_box)  # Manager will handle add with class_id

        logger.info(f"Added annotation with class_id: {class_id}")

    # Optional: Method to update classes if changed dynamically
    def update_classes(self, new_classes: List[str]):
        self.classes = new_classes
        self.class_combo.clear()
        self.class_combo.addItems(self.classes or ['Class 0'])