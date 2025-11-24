import napari
import numpy as np
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage
from scipy.ndimage import binary_erosion
from skimage.segmentation import expand_labels

from .layer_manager import LayerManager
from .process_actions_helpers import process_action_seg


def erode_labels(img: np.ndarray, diam: int, iterations: int) -> np.ndarray:
    """Erode labels with provided diameter, for given number of iterations."""

    structuring_element = (
        np.ones((diam, diam), dtype=bool)
        if img.ndim == 2
        else np.ones((diam, diam, diam), dtype=bool)
    )

    mask = img > 0
    filled_mask = ndimage.binary_fill_holes(mask)
    eroded_mask = binary_erosion(
        filled_mask,
        structure=structuring_element,
        iterations=iterations,
    )
    return np.where(eroded_mask, img, 0)


def expand_labels_skimage(img: np.ndarray, diam: int, iterations: int) -> np.ndarray:
    "Expand labels with given diameter, for given number of iterations."

    for _j in range(iterations):
        expanded_labels = expand_labels(img, distance=diam)

    return expanded_labels


class ErosionDilationWidget(QWidget):
    """Widget to perform erosion/dilation on label images"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        dil_erode_box = QGroupBox("Erode/dilate labels")
        dil_erode_box_layout = QVBoxLayout()

        radius_layout = QHBoxLayout()
        str_element_diameter_label = QLabel("Structuring element diameter")
        str_element_diameter_label.setFixedWidth(200)
        self.structuring_element_diameter = QSpinBox()
        self.structuring_element_diameter.setMaximum(100)
        self.structuring_element_diameter.setValue(1)
        radius_layout.addWidget(str_element_diameter_label)
        radius_layout.addWidget(self.structuring_element_diameter)

        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Iterations")
        iterations_label.setFixedWidth(200)
        self.iterations = QSpinBox()
        self.iterations.setMaximum(100)
        self.iterations.setValue(1)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations)

        shrink_dilate_buttons_layout = QHBoxLayout()
        self.erode_btn = QPushButton("Erode")
        self.dilate_btn = QPushButton("Dilate")
        self.erode_btn.clicked.connect(self._erode_labels)
        self.dilate_btn.clicked.connect(self._dilate_labels)
        shrink_dilate_buttons_layout.addWidget(self.erode_btn)
        shrink_dilate_buttons_layout.addWidget(self.dilate_btn)

        self.erode_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        self.label_manager.layer_update.connect(
            lambda: self.erode_btn.setEnabled(
                isinstance(self.label_manager.selected_layer, napari.layers.Labels)
            )
        )
        self.dilate_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        self.label_manager.layer_update.connect(
            lambda: self.dilate_btn.setEnabled(
                isinstance(self.label_manager.selected_layer, napari.layers.Labels)
            )
        )

        dil_erode_box_layout.addLayout(radius_layout)
        dil_erode_box_layout.addLayout(iterations_layout)
        dil_erode_box_layout.addLayout(shrink_dilate_buttons_layout)

        dil_erode_box.setLayout(dil_erode_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(dil_erode_box)
        self.setLayout(layout)

    def _erode_labels(self):
        """Shrink oversized labels through erosion"""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()
        action = erode_labels
        eroded = process_action_seg(
            seg=self.label_manager.selected_layer.data,
            action=action,
            basename=self.label_manager.selected_layer.name,
            diam=diam,
            iterations=iterations,
        )
        self.label_manager.selected_layer = self.viewer.add_labels(
            eroded,
            name=self.label_manager.selected_layer.name + "_eroded",
            scale=self.label_manager.selected_layer.scale,
        )

    def _dilate_labels(self):
        """Dilate labels in the selected layer."""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()

        action = expand_labels_skimage
        expanded = process_action_seg(
            seg=self.label_manager.selected_layer.data,
            action=action,
            basename=self.label_manager.selected_layer.name,
            diam=diam,
            iterations=iterations,
        )
        self.label_manager.selected_layer = self.viewer.add_labels(
            expanded,
            name=self.label_manager.selected_layer.name + "_expanded",
            scale=self.label_manager.selected_layer.scale,
        )
