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

        dim_box = QGroupBox("Dimensions")
        self.grid = QGridLayout()

        # headers
        self.grid.addWidget(QLabel("Index"), 0, 1)
        self.grid.addWidget(QLabel("Axis"), 0, 0)
        self.grid.addWidget(QLabel("Pixel scaling"), 0, 2)

        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_dims)

        dim_layout = QVBoxLayout()
        dim_layout.addLayout(self.grid)
        dim_layout.addWidget(self.apply_btn)
        dim_box.setLayout(dim_layout)

        layout = QVBoxLayout()
        layout.addWidget(dim_box)
        self.setLayout(layout)

        # Connect to viewer signal to update the active layer.
        self.viewer.layers.selection.events.changed.connect(self._on_selection_changed)
        self.populate_dimensions(self.layer)

    def _on_selection_changed(self) -> None:
        """Update the active layer"""

        if (
            len(self.viewer.layers.selection) == 1
        ):  # Only consider single layer selection
            selected_layer = self.viewer.layers.selection.active
            if isinstance(selected_layer, napari.layers.Labels | napari.layers.Image):
                self.layer = selected_layer
            else:
                self.layer = None
                return

        self.populate_dimensions(self.layer)

    def populate_dimensions(self, layer: napari.layers.Layer | None) -> None:
        """Populate the dimension information based on the given layer"""

        # Clear existing widgets
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget is not None and widget not in [self.apply_btn]:
                self.grid.removeWidget(widget)
                widget.deleteLater()

        self.layer = layer
        self.axis_widgets = []

        if layer is None:
            for widget in self.axis_widgets:
                for w in widget:
                    w.setVisible(False)
            return

        # Check for existing dimension information in the layer metadata
        if "dimension_info" in layer.metadata:
            dims, axes_labels, scale_info = layer.metadata["dimension_info"]
            offset = 0
        else:
            ndim = layer.data.ndim
            axes_labels = ["C", "T", "Z", "Y", "X"]
            dims = range(ndim)
            scale_info = [1.0] * ndim
            offset = len(axes_labels) - ndim

        for d, label in enumerate(dims):
            dim = QLabel(str(label))
            dim.setVisible(True)
            axis = QComboBox()
            axis.addItems(["C", "T", "Z", "Y", "X"])
            axis.setCurrentText(axes_labels[d + offset])
            axis.setVisible(True)
            scale = QDoubleSpinBox()
            scale.setValue(scale_info[d])
            scale.setSingleStep(0.1)
            scale.setMinimum(0.01)

            if axis.currentText() in ("C", "T"):
                scale.setVisible(False)
            else:
                scale.setVisible(True)

            axis.currentIndexChanged.connect(
                lambda _, a=axis, s=scale: s.setVisible(
                    a.currentText() not in ("C", "T")
                )
            )
            axis.currentIndexChanged.connect(self.update_apply_button_state)

            self.grid.addWidget(dim, d + 1, 0)
            self.grid.addWidget(axis, d + 1, 1)
            self.grid.addWidget(scale, d + 1, 2)

            self.axis_widgets.append((dim, axis, scale))

        self.apply_dims()

    def update_apply_button_state(self) -> None:
        """Update whether the apply button should be enabled"""

        # check for duplicate axes
        axes = [
            self.axis_widgets[i][1].currentText() for i in range(len(self.axis_widgets))
        ]
        if len(axes) != len(set(axes)):
            self.apply_btn.setEnabled(False)
            return

        # check that Y and X are present
        if "Y" not in axes or "X" not in axes:
            self.apply_btn.setEnabled(False)
            return

        self.apply_btn.setEnabled(True)

    def apply_dims(self) -> None:
        """Apply the dimension information to the selected layer"""

        if self.layer is None:
            return

        _, axes, scale = self.get_dimension_info()

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
        self.layer.data = np.transpose(self.layer.data, transpose_order)

        self.layer.metadata["dimension_info"] = self.get_dimension_info()
        self.dims_updated.emit()

    def get_dimension_info(self) -> list[tuple[str, float]]:
        """Get the dimension info from the widget"""

        dims = [self.axis_widgets[i][0].text() for i in range(len(self.axis_widgets))]
        axes = [
            self.axis_widgets[i][1].currentText() for i in range(len(self.axis_widgets))
        ]
        scale = [self.axis_widgets[i][2].value() for i in range(len(self.axis_widgets))]

        return dims, axes, scale
