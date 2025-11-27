import napari
import numpy as np
from PyQt5.QtCore import pyqtSignal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DimensionWidget(QWidget):
    """QWidget to display and edit dimension information."""

    dims_updated = pyqtSignal()  # signal to emit when dims change

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.layer = None
        self.row_active = [False] * 5

        dim_box = QGroupBox("Dimensions")
        label = QLabel("Dimensions are (re)ordered in CTZYX order")
        label.setWordWrap(True)
        font = label.font()
        font.setItalic(True)
        label.setFont(font)

        self.grid = QGridLayout()

        # headers
        self.grid.addWidget(QLabel("Axis"), 0, 0)
        self.grid.addWidget(QLabel("Name"), 0, 1)
        self.grid.addWidget(QLabel("Pixel scaling"), 0, 2)

        # Pre-create all widget rows
        self.axis_widgets = []
        for row in range(5):
            dim_label = QLabel("")
            axis_combo = QComboBox()
            axis_combo.addItems(["C", "T", "Z", "Y", "X"])

            scale_spin = QDoubleSpinBox()
            scale_spin.setSingleStep(0.1)
            scale_spin.setMinimum(0.01)

            axis_combo.currentIndexChanged.connect(
                lambda _, a=axis_combo, s=scale_spin: s.setVisible(
                    a.currentText() not in ("C", "T")
                )
            )
            axis_combo.currentIndexChanged.connect(self.update_apply_button_state)

            # Add to layout
            self.grid.addWidget(dim_label, row + 1, 0)
            self.grid.addWidget(axis_combo, row + 1, 1)
            self.grid.addWidget(scale_spin, row + 1, 2)

            # hide for now
            dim_label.hide()
            axis_combo.hide()
            scale_spin.hide()

            # Store tuple
            self.axis_widgets.append((dim_label, axis_combo, scale_spin))

        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_dims)

        dim_layout = QVBoxLayout()
        dim_layout.addWidget(label)
        dim_layout.addLayout(self.grid)
        dim_layout.addWidget(self.apply_btn)
        dim_box.setLayout(dim_layout)

        layout = QVBoxLayout()
        layout.addWidget(dim_box)
        self.setLayout(layout)

        # Connect to viewer signal to update the active layer.
        self.viewer.layers.selection.events.active.connect(self._on_selection_changed)

    def _on_selection_changed(self, event=None) -> None:
        """Update the active layer"""

        if (
            len(self.viewer.layers.selection) == 1
        ):  # Only consider single layer selection
            selected_layer = self.viewer.layers.selection.active
            if isinstance(selected_layer, napari.layers.Labels | napari.layers.Image):
                self.layer = selected_layer
                self._update_dimensions()
            else:
                self.layer = None

    def _update_dimensions(self) -> None:
        """Update the dimension names, order, and scaling. Dimensions are always ordered
        in CTZYX order."""

        ndim = self.layer.data.ndim

        if "dimension_info" in self.layer.metadata:
            _, axes_labels, _ = self.layer.metadata["dimension_info"]
            offset = 0
        else:
            axes_labels = ["C", "T", "Z", "Y", "X"]
            offset = len(axes_labels) - ndim

        scale_info = self.layer.scale

        # Enable/disable rows
        for i, (dim_label, axis_combo, scale_spin) in enumerate(self.axis_widgets):
            if i < ndim:
                self.row_active[i] = True

                # Show needed rows
                dim_label.show()
                axis_combo.show()

                axis_combo.setCurrentText(axes_labels[i + offset])
                dim_label.setText(str(i) + f" [{self.layer.data.shape[i]}]")

                # Set scale
                s = scale_info[i] if i < len(scale_info) else 1
                scale_spin.setValue(s)
                scale_spin.setVisible(axis_combo.currentText() not in ("C", "T"))
            else:
                self.row_active[i] = False

                # Hide unused rows
                dim_label.hide()
                axis_combo.hide()
                scale_spin.hide()

        if "dimension_info" not in self.layer.metadata:
            self.apply_dims()
        else:
            self.update_apply_button_state()
            self.dims_updated.emit()

    def update_apply_button_state(self) -> None:
        """Check if the current dimensions are valid (must include Y,X, no duplicate axes)"""

        axes = [
            axis_combo.currentText()
            for active, (dim_label, axis_combo, scale_spin) in zip(
                self.row_active, self.axis_widgets, strict=False
            )
            if active
        ]

        # Duplicate axes?
        if len(axes) != len(set(axes)):
            self.apply_btn.setEnabled(False)
            return

        # Must contain Y and X
        if "Y" not in axes or "X" not in axes:
            self.apply_btn.setEnabled(False)
            return

        self.apply_btn.setEnabled(True)

    def apply_dims(self) -> None:
        """Apply the dimension information to the selected layer"""

        if self.layer is None:
            return

        dims, axes, scale = self.get_dimension_info()

        # update layer and viewer dims scale
        old_step = self.viewer.dims.current_step

        # set scale according to widget values. Channels and Time should always have scale 1
        scale = [
            1 if axis in ("C", "T") else s for axis, s in zip(axes, scale, strict=False)
        ]

        self.layer.scale = tuple(scale)
        self.viewer.reset_view()
        self.viewer.dims.current_step = old_step

        # transpose if necessary
        napari_order = ["C", "T", "Z", "Y", "X"]
        current_order = axes
        transpose_order = []
        for axis in napari_order:
            if axis in current_order:
                transpose_order.append(current_order.index(axis))
        if transpose_order != list(range(len(transpose_order))):
            self.layer.data = np.transpose(self.layer.data, transpose_order)

        self.layer.metadata["dimension_info"] = (dims, axes, scale)
        self._update_dimensions()  # to update text in the widgets

    def get_dimension_info(self) -> None:
        """Extract dimension info from the current settings in the widgets"""

        dims, axes, scale = [], [], []

        for active, (dim_label, axis_combo, scale_spin) in zip(
            self.row_active, self.axis_widgets, strict=False
        ):
            if active:
                dims.append(dim_label.text())
                axes.append(axis_combo.currentText())
                scale.append(scale_spin.value())

        return dims, axes, scale
