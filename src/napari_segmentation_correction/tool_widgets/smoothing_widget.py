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

from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action_seg,
)
from napari_segmentation_correction.tool_widgets.base_tool_widget import BaseToolWidget


def median_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Return the resulting image after applying median filter of given size."""

    return ndimage.median_filter(img, size=size)


class SmoothingWidget(BaseToolWidget):
    """Widget that 'smooths' labels by applying a median filter"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(napari.layers.Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        box = QGroupBox("Smooth objects")
        box_layout = QVBoxLayout()

        self.median_radius_field = QSpinBox()
        self.median_radius_field.setMaximum(100)
        self.smooth_btn = QPushButton("Smooth")
        self.smooth_btn.clicked.connect(self._smooth_objects)
        self.smooth_btn.setEnabled(self.layer is not None)
        self.update_status.connect(
            lambda: self.smooth_btn.setEnabled(self.layer is not None)
        )

        widget_layout = QHBoxLayout()
        widget_layout.addWidget(self.median_radius_field)
        widget_layout.addWidget(self.smooth_btn)

        box_layout.addWidget(QLabel("Median filter radius"))
        box_layout.addLayout(widget_layout)

        box.setLayout(box_layout)
        layout = QVBoxLayout()
        layout.addWidget(box)
        self.setLayout(layout)

    def _smooth_objects(self) -> None:
        """Smooth objects by using a median filter."""

        action = median_filter
        smoothed = process_action_seg(
            self.layer.data,
            action,
            basename=self.layer.name,
            size=self.median_radius_field.value(),
        )
        if smoothed is not None:
            self.layer = self.viewer.add_labels(
                smoothed,
                name=self.layer.name + "_medianfilter",
                scale=self.layer.scale,
            )
