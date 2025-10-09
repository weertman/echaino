# FILE: ui\manage_classes_dialog.py
# PATH: D:\urchinScanner\ui\manage_classes_dialog.py

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLineEdit, QLabel, QMessageBox, QDialogButtonBox
)
from PyQt6.QtCore import Qt

class ManageClassesDialog(QDialog):
    """
    Simple dialog to add/remove/rename/reorder class names.
    Colors are assigned in ControlPanel after accept(), not here.
    """
    def __init__(self, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Classes")
        self.resize(420, 360)
        self._list = QListWidget()
        self._list.setSelectionMode(self._list.SelectionMode.SingleSelection)
        self._list.setDragDropMode(self._list.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)

        for name in class_names or []:
            self._list.addItem(QListWidgetItem(str(name)))

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("New class name…")

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_current_text)

        rename_btn = QPushButton("Rename Selected")
        rename_btn.clicked.connect(self._rename_selected)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)

        up_btn = QPushButton("Move Up")
        up_btn.clicked.connect(self._move_up)

        down_btn = QPushButton("Move Down")
        down_btn.clicked.connect(self._move_down)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Classes (drag to reorder):"))
        layout.addWidget(self._list)

        row = QHBoxLayout()
        row.addWidget(self._name_edit, stretch=1)
        row.addWidget(add_btn)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(rename_btn)
        row2.addWidget(remove_btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(up_btn)
        row3.addWidget(down_btn)
        layout.addLayout(row3)

        layout.addWidget(buttons)

    def _add_current_text(self):
        text = self._name_edit.text().strip()
        if not text:
            return
        # prevent exact duplicates
        names = [self._list.item(i).text() for i in range(self._list.count())]
        if text in names:
            QMessageBox.warning(self, "Duplicate", f'"{text}" already exists.')
            return
        self._list.addItem(QListWidgetItem(text))
        self._name_edit.clear()

    def _rename_selected(self):
        it = self._list.currentItem()
        if not it:
            return
        text = self._name_edit.text().strip()
        if not text:
            return
        names = [self._list.item(i).text() for i in range(self._list.count())]
        if text in names and text != it.text():
            QMessageBox.warning(self, "Duplicate", f'"{text}" already exists.')
            return
        it.setText(text)
        self._name_edit.clear()

    def _remove_selected(self):
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)

    def _move_up(self):
        row = self._list.currentRow()
        if row > 0:
            it = self._list.takeItem(row)
            self._list.insertItem(row - 1, it)
            self._list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self._list.currentRow()
        if row >= 0 and row < self._list.count() - 1:
            it = self._list.takeItem(row)
            self._list.insertItem(row + 1, it)
            self._list.setCurrentRow(row + 1)

    def classes(self):
        return [self._list.item(i).text() for i in range(self._list.count())]
