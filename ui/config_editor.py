# FILE: ui\config_editor.py
# PATH: D:\\echaino\\ui\config_editor.py

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QGroupBox, QPushButton, QScrollArea, QWidget, QMessageBox
from PyQt6.QtCore import Qt
from typing import Dict, Any, Optional
import copy  # For deep copy of config

class ConfigEditorDialog(QDialog):
    """Dynamic dialog to edit config dict for project-specific overrides."""

    def __init__(self, root_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Project Configuration")
        self.setMinimumSize(600, 400)  # Reasonable size

        self.root_config = copy.deepcopy(root_config)  # Preserve original
        self.edited_config = copy.deepcopy(root_config)  # Working copy

        # Main layout with scroll for large configs
        main_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.form_layout = QFormLayout(container)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Build form recursively
        self.widgets = {}  # Map: key_path -> widget for value collection
        self._build_form(self.edited_config, self.form_layout, key_prefix='')

        # Buttons
        button_layout = QVBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset)
        button_layout.addWidget(reset_btn)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _build_form(self, config: Dict[str, Any], layout: QFormLayout, key_prefix: str):
        """Recursively add fields for config items."""
        for key, value in config.items():
            full_key = f"{key_prefix}.{key}" if key_prefix else key
            if isinstance(value, dict):
                # Nested: GroupBox
                group = QGroupBox(key.capitalize())
                group_layout = QFormLayout()
                self._build_form(value, group_layout, full_key)
                group.setLayout(group_layout)
                layout.addRow(group)
            else:
                # Leaf: Appropriate widget
                widget = self._create_widget(value, full_key)
                if widget:
                    layout.addRow(key.capitalize(), widget)
                    self.widgets[full_key] = widget

    def _create_widget(self, value: Any, full_key: str) -> Optional[QWidget]:
        """Create type-appropriate widget."""
        if isinstance(value, bool):
            chk = QCheckBox()
            chk.setChecked(value)
            return chk
        elif isinstance(value, int):
            spin = QSpinBox()
            spin.setValue(value)
            spin.setMinimum(0)  # Example validation; customize per key if needed
            return spin
        elif isinstance(value, float):
            dbl_spin = QDoubleSpinBox()
            dbl_spin.setValue(value)
            dbl_spin.setMinimum(0.0)
            return dbl_spin
        elif isinstance(value, str):
            edit = QLineEdit()
            edit.setText(value)
            return edit
        elif isinstance(value, list):
            if all(isinstance(v, (int, float, str)) for v in value):  # Simple list: Combo or edit
                if len(value) <= 10:  # Combo for small lists
                    combo = QComboBox()
                    combo.addItems([str(v) for v in value])
                    return combo
                else:  # Large: Comma-separated edit
                    edit = QLineEdit()
                    edit.setText(','.join(map(str, value)))
                    return edit
            else:
                # Complex list: Skip or handle as str (warn)
                print(f"Warning: Complex list for {full_key}; editing as str")
                edit = QLineEdit()
                edit.setText(str(value))
                return edit
        else:
            # Unsupported: Read-only label
            print(f"Warning: Unsupported type for {full_key}: {type(value)}")
            return None

    def _collect_values(self) -> Dict[str, Any]:
        """Recurse and collect widget values back into dict."""
        def set_nested(d: Dict, keys: List[str], val: Any):
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

        collected = copy.deepcopy(self.root_config)  # Start from root structure
        for full_key, widget in self.widgets.items():
            keys = full_key.split('.')
            val = self._get_widget_value(widget)
            if val is not None:
                set_nested(collected, keys, val)
        return collected

    def _get_widget_value(self, widget: QWidget) -> Any:
        """Extract value from widget with type conversion."""
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QLineEdit):
            text = widget.text()
            # Try int/float/list parse
            try:
                return int(text)
            except ValueError:
                try:
                    return float(text)
                except ValueError:
                    if ',' in text:  # List attempt
                        return [v.strip() for v in text.split(',')]
                    return text
        elif isinstance(widget, QComboBox):
            return widget.currentText()
        return None

    def _reset(self):
        """Reset widgets to root values."""
        for full_key, widget in self.widgets.items():
            keys = full_key.split('.')
            val = self.root_config
            for k in keys:
                val = val.get(k, None)
            if val is not None:
                self._set_widget_value(widget, val)

    def _set_widget_value(self, widget: QWidget, value: Any):
        if isinstance(widget, QCheckBox):
            widget.setChecked(value)
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value) if not isinstance(value, list) else ','.join(map(str, value)))
        elif isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)

    def accept(self):
        """Validate and collect on OK."""
        # Basic validation example (expand as needed)
        for full_key, widget in self.widgets.items():
            if isinstance(widget, QSpinBox) and widget.value() < 0:
                QMessageBox.warning(self, "Validation Error", f"Value for {full_key} must be >= 0")
                return
        self.edited_config = self._collect_values()
        super().accept()

    def get_edited_config(self) -> Dict[str, Any]:
        return self.edited_config
