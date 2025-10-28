# FILE: ui/cross_archive_analysis_dialog.py
# PATH: D:\echaino\ui\cross_archive_analysis_dialog.py

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QCheckBox, QLabel,
    QDialogButtonBox, QAbstractItemView
)
from typing import Dict, List


class CrossArchiveAnalysisDialog(QDialog):
    """
    Cross-Archive Analysis configuration dialog.

    Changes:
      - Replaces single 'Diameter' checkbox with two explicit options:
          * Diameter (best-fit circle)
          * Diameter (Feret max, tip-to-tip)
      - Returns metric keys compatible with CrossArchiveAnalyzer.um_metrics_map:
          'area', 'perimeter', 'diameter_best_fit', 'diameter_feret_max'
    """

    def __init__(self, config: Dict, archives: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cross-Archive Analysis")

        layout = QVBoxLayout(self)

        # Archive selection list
        self.archive_list = QListWidget()
        self.archive_list.addItems(archives or [])
        self.archive_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(QLabel("Select Archives to Compare:"))
        layout.addWidget(self.archive_list)

        # Global options
        self.auto_generate_cb = QCheckBox("Automatically Generate Missing measurements.csv")
        self.auto_generate_cb.setChecked(True)
        layout.addWidget(self.auto_generate_cb)

        # Metric selection
        layout.addWidget(QLabel("Metrics to Include:"))

        self.area_cb = QCheckBox("Area")
        self.area_cb.setChecked(True)
        layout.addWidget(self.area_cb)

        self.perimeter_cb = QCheckBox("Perimeter")
        self.perimeter_cb.setChecked(True)
        layout.addWidget(self.perimeter_cb)

        # NEW: explicit diameter choices
        self.diam_best_cb = QCheckBox("Diameter (best-fit circle)")
        self.diam_best_cb.setChecked(True)
        layout.addWidget(self.diam_best_cb)

        self.diam_feret_cb = QCheckBox("Diameter (Feret max, tip-to-tip)")
        self.diam_feret_cb.setChecked(True)
        layout.addWidget(self.diam_feret_cb)

        # OK/Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    # Public API -------------------------------------------------------------

    def get_selected_archives(self) -> List[str]:
        return [item.text() for item in self.archive_list.selectedItems()]

    def get_params(self) -> Dict:
        """
        Returns:
          {
            'auto_generate': bool,
            'metrics': List[str]  # compatible with CrossArchiveAnalyzer.um_metrics_map
          }
        """
        metrics: List[str] = []
        if self.area_cb.isChecked():
            metrics.append('area')
        if self.perimeter_cb.isChecked():
            metrics.append('perimeter')
        if self.diam_best_cb.isChecked():
            metrics.append('diameter_best_fit')
        if self.diam_feret_cb.isChecked():
            metrics.append('diameter_feret_max')

        return {
            'auto_generate': self.auto_generate_cb.isChecked(),
            'metrics': metrics
        }
