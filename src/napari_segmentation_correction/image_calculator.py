
import dask.array as da
import napari
import numpy as np
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._layer_dropdown import LayerDropdown


class ImageCalculator(QWidget):
    """Widget to perform calculations between two images"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer

        ### Add one image to another
        image_calc_box = QGroupBox("Image Calculator")
        image_calc_box_layout = QVBoxLayout()

        image1_layout = QHBoxLayout()
        image1_layout.addWidget(QLabel("Label image 1"))
        self.image1_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image1_dropdown.layer_changed.connect(self._update_image1)
        image1_layout.addWidget(self.image1_dropdown)

        image2_layout = QHBoxLayout()
        image2_layout.addWidget(QLabel("Label image 2"))
        self.image2_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image2_dropdown.layer_changed.connect(self._update_image2)
        image2_layout.addWidget(self.image2_dropdown)

        image_calc_box_layout.addLayout(image1_layout)
        image_calc_box_layout.addLayout(image2_layout)

        operation_layout = QHBoxLayout()
        self.operation = QComboBox()
        self.operation.addItem("Add")
        self.operation.addItem("Subtract")
        self.operation.addItem("Multiply")
        self.operation.addItem("Divide")
        self.operation.addItem("AND")
        self.operation.addItem("OR")
        operation_layout.addWidget(QLabel("Operation"))
        operation_layout.addWidget(self.operation)
        image_calc_box_layout.addLayout(operation_layout)

        add_images_btn = QPushButton("Run")
        add_images_btn.clicked.connect(self._calculate_images)
        image_calc_box_layout.addWidget(add_images_btn)

        image_calc_box.setLayout(image_calc_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(image_calc_box)
        self.setLayout(main_layout)

    def _update_image1(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

    def _update_image2(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image2_layer = None
        else:
            self.image2_layer = self.viewer.layers[selected_layer]
            self.image2_dropdown.setCurrentText(selected_layer)

    def _calculate_images(self):
        """Add label image 2 to label image 1"""

        if (
            isinstance(self.image1_layer, da.core.Array)
            or isinstance(self.image2_layer, da.core.Array)
        ):
            msg = QMessageBox()
            msg.setWindowTitle(
                "Cannot yet run image calculator on dask arrays"
            )
            msg.setText("Cannot yet run image calculator on dask arrays")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        if self.image1_layer.data.shape != self.image2_layer.data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Images must have the same shape")
            msg.setText("Images must have the same shape")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        if self.operation.currentText() == "Add":
            self.viewer.add_image(
                np.add(self.image1_layer.data, self.image2_layer.data)
            )
        if self.operation.currentText() == "Subtract":
            self.viewer.add_image(
                np.subtract(self.image1_layer.data, self.image2_layer.data)
            )
        if self.operation.currentText() == "Multiply":
            self.viewer.add_image(
                np.multiply(self.image1_layer.data, self.image2_layer.data)
            )
        if self.operation.currentText() == "Divide":
            self.viewer.add_image(
                np.divide(
                    self.image1_layer.data,
                    self.image2_layer.data,
                    out=np.zeros_like(self.image1_layer.data, dtype=float),
                    where=self.image2_layer.data != 0,
                )
            )
        if self.operation.currentText() == "AND":
            self.viewer.add_labels(
                np.logical_and(
                    self.image1_layer.data != 0, self.image2_layer.data != 0
                ).astype(int)
            )
        if self.operation.currentText() == "OR":
            self.viewer.add_labels(
                np.logical_or(
                    self.image1_layer.data != 0, self.image2_layer.data != 0
                ).astype(int)
            )
