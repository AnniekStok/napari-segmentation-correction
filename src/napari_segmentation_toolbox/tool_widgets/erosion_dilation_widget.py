import napari
import numpy as np
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from scipy import ndimage
from scipy.ndimage import binary_erosion
from skimage.segmentation import expand_labels

from napari_segmentation_toolbox.helpers.base_tool_widget import BaseToolWidget
from napari_segmentation_toolbox.helpers.process_actions_helpers import (
    process_action_seg,
)


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


class ErosionDilationWidget(BaseToolWidget):
    """Widget to perform erosion/dilation on label images"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(napari.layers.Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        box = QGroupBox("Erode/dilate labels")
        box_layout = QVBoxLayout()

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

        erode_expand_buttons_layout = QHBoxLayout()
        self.erode_btn = QPushButton("Erode")
        self.dilate_btn = QPushButton("Dilate")
        self.erode_btn.clicked.connect(lambda: self._erode_dilate_labels(erode=True))
        self.dilate_btn.clicked.connect(lambda: self._erode_dilate_labels(erode=False))
        erode_expand_buttons_layout.addWidget(self.erode_btn)
        erode_expand_buttons_layout.addWidget(self.dilate_btn)

        self.erode_btn.setEnabled(self.layer is not None)
        self.dilate_btn.setEnabled(self.layer is not None)
        self.update_status.connect(self._update_buttons)

        box_layout.addLayout(radius_layout)
        box_layout.addLayout(iterations_layout)
        box_layout.addLayout(erode_expand_buttons_layout)

        box.setLayout(box_layout)

        layout = QVBoxLayout()
        layout.addWidget(box)
        self.setLayout(layout)

    def _update_buttons(self):
        """Update button state"""

        enabled = self.layer is not None
        self.erode_btn.setEnabled(enabled)
        self.dilate_btn.setEnabled(enabled)

    def _erode_dilate_labels(self, erode: bool) -> None:
        """Shrink labels through erosion or dilate by expansion"""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()

        if erode:
            name = self.layer.name + "_eroded"
            action = erode_labels
        else:
            name = self.layer.name + "_dilated"
            action = expand_labels_skimage

        result = process_action_seg(
            seg=self.layer.data,
            action=action,
            basename=self.layer.name,
            diam=diam,
            iterations=iterations,
        )

        if result is not None:
            self.layer = self.viewer.add_labels(
                result,
                name=name,
                scale=self.layer.scale,
            )
