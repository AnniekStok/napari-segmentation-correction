import napari
import numpy as np
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_segmentation_correction.helpers.layer_dropdown import LayerDropdown
from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action_seg,
)


def threshold(
    img: np.ndarray, min_val: int | float, max_val: int | float
) -> np.ndarray:
    """Threshold the input image"""

    return (img >= min_val) & (img <= max_val)


class ThresholdWidget(QWidget):
    """Widget that applies a threshold to an image or labels layer"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer
        self.outputdir = None
        self.threshold_layer = None

        threshold_box = QGroupBox("Threshold")
        threshold_box_layout = QVBoxLayout()

        self.threshold_layer_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.threshold_layer_dropdown.layer_changed.connect(
            self._update_threshold_layer
        )
        threshold_box_layout.addWidget(self.threshold_layer_dropdown)

        min_threshold_layout = QHBoxLayout()
        min_threshold_layout.addWidget(QLabel("Min value"))
        self.min_threshold = QSpinBox()
        self.min_threshold.setMaximum(65535)
        min_threshold_layout.addWidget(self.min_threshold)

        max_threshold_layout = QHBoxLayout()
        max_threshold_layout.addWidget(QLabel("Max value"))
        self.max_threshold = QSpinBox()
        self.max_threshold.setMaximum(65535)
        self.max_threshold.setValue(65535)
        max_threshold_layout.addWidget(self.max_threshold)

        threshold_box_layout.addLayout(min_threshold_layout)
        threshold_box_layout.addLayout(max_threshold_layout)
        threshold_btn = QPushButton("Run")
        threshold_btn.clicked.connect(self._threshold)

        threshold_btn.setEnabled(
            isinstance(
                self.threshold_layer_dropdown.selected_layer,
                napari.layers.Labels | napari.layers.Image,
            )
        )
        self.threshold_layer_dropdown.layer_changed.connect(
            lambda: threshold_btn.setEnabled(
                isinstance(
                    self.threshold_layer_dropdown.selected_layer,
                    napari.layers.Labels | napari.layers.Image,
                )
            )
        )

        threshold_box_layout.addWidget(threshold_btn)
        threshold_box.setLayout(threshold_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(threshold_box)
        self.setLayout(layout)

    def _update_threshold_layer(self, selected_layer) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.threshold_layer = None
        else:
            self.threshold_layer = self.viewer.layers[selected_layer]
            self.threshold_layer_dropdown.setCurrentText(selected_layer)

    def _threshold(self):
        """Threshold the selected label or intensity image"""

        action = threshold
        mask = process_action_seg(
            seg=self.threshold_layer.data,
            action=action,
            basename=self.threshold_layer.name,
            min_val=self.min_threshold.value(),
            max_val=self.max_threshold.value(),
        )
        if mask is not None:
            self.viewer.add_labels(
                mask,
                name=self.threshold_layer.name + "_threshold",
                scale=self.threshold_layer.scale,
            )
