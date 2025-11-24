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

from napari_segmentation_correction.process_actions_helpers import process_action

from .layer_dropdown import LayerDropdown


def add_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.add(img1, img2)


def subtract_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.subtract(img1, img2)


def multiply_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.multiply(img1, img2)


def divide_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.divide(img1, img2, out=np.zeros_like(img1, dtype=float), where=img2 != 0)


def logical_and(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.logical_and(img1 != 0, img2 != 0).astype(int)


def logical_or(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.logical_or(img1 != 0, img2 != 0).astype(int)


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
        add_images_btn.setEnabled(
            self.image1_dropdown.selected_layer is not None
            and self.image2_dropdown.selected_layer is not None
        )
        self.image1_dropdown.layer_changed.connect(
            lambda: add_images_btn.setEnabled(
                self.image1_dropdown.selected_layer is not None
                and self.image2_dropdown.selected_layer is not None
            )
        )
        self.image2_dropdown.layer_changed.connect(
            lambda: add_images_btn.setEnabled(
                self.image1_dropdown.selected_layer is not None
                and self.image2_dropdown.selected_layer is not None
            )
        )

        image_calc_box_layout.addWidget(add_images_btn)

        image_calc_box.setLayout(image_calc_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(image_calc_box)
        self.setLayout(main_layout)

    def _update_image1(self, selected_layer: str) -> None:
        """Update the layer for image 1"""

        if selected_layer == "":
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

    def _update_image2(self, selected_layer: str) -> None:
        """Update the layer for image 2"""

        if selected_layer == "":
            self.image2_layer = None
        else:
            self.image2_layer = self.viewer.layers[selected_layer]
            self.image2_dropdown.setCurrentText(selected_layer)

    def _calculate_images(self):
        """Execute mathematical operations between two images."""

        if self.image1_layer.data.shape != self.image2_layer.data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Images must have the same shape")
            msg.setText("Images must have the same shape")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        if self.operation.currentText() == "Add":
            action = add_images
        elif self.operation.currentText() == "Subtract":
            action = subtract_images
        elif self.operation.currentText() == "Multiply":
            action = multiply_images
        elif self.operation.currentText() == "Divide":
            action = divide_images
        elif self.operation.currentText() == "AND":
            action = logical_and
        elif self.operation.currentText() == "OR":
            action = logical_or

        if isinstance(self.image1_layer.data, da.core.Array) or isinstance(
            self.image2_layer.data, da.core.Array
        ):
            indices = range(self.image1_layer.data.shape[0])
            result = process_action(
                self.image1_layer.data,
                self.image2_layer.data,
                action,
                basename=self.image1_layer.name,
                img1_index=indices,
                img2_index=indices,
            )
        else:
            result = process_action(
                self.image1_layer.data,
                self.image2_layer.data,
                action,
                basename=self.image1_layer.name,
            )
        self.viewer.add_image(
            result,
            name=f"{self.image1_layer.name}_{self.image2_layer.name}_{self.operation.currentText()}",
            scale=self.image1_layer.scale,
        )
