# FILE: ui/cross_archive_analysis_dialog.py

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QCheckBox, QLabel, QDialogButtonBox, QAbstractItemView
from typing import Dict, List

class CrossArchiveAnalysisDialog(QDialog):
    def __init__(self, config: Dict, archives: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cross-Archive Analysis")
        layout = QVBoxLayout()

        self.archive_list = QListWidget()
        self.archive_list.addItems(archives)
        self.archive_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(QLabel("Select Archives to Compare:"))
        layout.addWidget(self.archive_list)

        self.auto_generate_cb = QCheckBox("Automatically Generate Missing measurements.csv")
        self.auto_generate_cb.setChecked(True)
        layout.addWidget(self.auto_generate_cb)

        layout.addWidget(QLabel("Metrics to Include:"))
        self.area_cb = QCheckBox("Area")
        self.area_cb.setChecked(True)
        layout.addWidget(self.area_cb)
        self.perimeter_cb = QCheckBox("Perimeter")
        self.perimeter_cb.setChecked(True)
        layout.addWidget(self.perimeter_cb)
        self.diameter_cb = QCheckBox("Diameter")
        self.diameter_cb.setChecked(True)
        layout.addWidget(self.diameter_cb)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selected_archives(self) -> List[str]:
        return [item.text() for item in self.archive_list.selectedItems()]

    def get_params(self) -> Dict:
        metrics = []
        if self.area_cb.isChecked():
            metrics.append('area')
        if self.perimeter_cb.isChecked():
            metrics.append('perimeter')
        if self.diameter_cb.isChecked():
            metrics.append('diameter')
        return {
            'auto_generate': self.auto_generate_cb.isChecked(),
            'metrics': metrics
        }