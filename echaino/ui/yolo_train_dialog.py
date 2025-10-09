from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QSpinBox, QPushButton, QLabel, QDialogButtonBox, QAbstractItemView, QDoubleSpinBox
from typing import Dict, List

class YoloTrainDialog(QDialog):
    def __init__(self, config: Dict, archives: List[str]):
        super().__init__()
        self.setWindowTitle("Train YOLO Model")
        self.config = config  # NEW: Store config for classes
        layout = QVBoxLayout()

        self.archive_list = QListWidget()
        self.archive_list.addItems(archives)
        self.archive_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(QLabel("Select Archives:"))
        layout.addWidget(self.archive_list)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(1000)
        self.epochs_spin.setValue(config['yolo'].get('train_epochs', 50))
        layout.addWidget(QLabel("Epochs:"))
        layout.addWidget(self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(128)
        self.batch_spin.setValue(config['yolo'].get('batch_size', 16))
        layout.addWidget(QLabel("Batch Size:"))
        layout.addWidget(self.batch_spin)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setMinimum(320)
        self.img_size_spin.setMaximum(1280)
        self.img_size_spin.setSingleStep(64)
        self.img_size_spin.setValue(config['yolo'].get('img_size', 640))
        layout.addWidget(QLabel("Image Size:"))
        layout.addWidget(self.img_size_spin)

        self.split_spin = QDoubleSpinBox()
        self.split_spin.setMinimum(0.5)
        self.split_spin.setMaximum(0.95)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setValue(config['yolo'].get('train_val_split', 0.8))
        layout.addWidget(QLabel("Train/Val Split:"))
        layout.addWidget(self.split_spin)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setMinimum(1)
        self.min_area_spin.setMaximum(1000)
        self.min_area_spin.setValue(config['yolo'].get('min_clip_area', 10))
        layout.addWidget(QLabel("Min Clip Area:"))
        layout.addWidget(self.min_area_spin)

        # NEW: Display classes (non-selectable list for info)
        self.classes_list = QListWidget()
        classes = config.get('classes', ['Class 0'])
        self.classes_list.addItems(classes)
        self.classes_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)  # Display only
        layout.addWidget(QLabel("Classes:"))
        layout.addWidget(self.classes_list)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selected_archives(self) -> List[str]:
        return [item.text() for item in self.archive_list.selectedItems()]

    def get_params(self) -> Dict:
        params = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'img_size': self.img_size_spin.value(),
            'train_val_split': self.split_spin.value(),
            'min_clip_area': self.min_area_spin.value()
        }
        # NEW: Add classes (full list; make selectable if filtering needed)
        params['classes'] = [self.classes_list.item(i).text() for i in range(self.classes_list.count())]
        return params