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

from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action_seg,
)
from napari_segmentation_correction.layer_control_widgets.layer_manager import (
    LayerManager,
)


def median_filter(img: np.ndarray, size: int) -> np.ndarray:
    return ndimage.median_filter(img, size=size)


class SmoothingWidget(QWidget):
    """Widget that 'smooths' labels by applying a median filter"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        smoothbox = QGroupBox("Smooth objects")
        smooth_boxlayout = QVBoxLayout()

        smooth_layout = QHBoxLayout()
        self.median_radius_field = QSpinBox()
        self.median_radius_field.setMaximum(100)
        self.smooth_btn = QPushButton("Smooth")
        smooth_layout.addWidget(self.median_radius_field)
        smooth_layout.addWidget(self.smooth_btn)

        smooth_boxlayout.addWidget(QLabel("Median filter radius"))
        smooth_boxlayout.addLayout(smooth_layout)

        self.smooth_btn.clicked.connect(self._smooth_objects)
        self.smooth_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        self.label_manager.layer_update.connect(
            lambda: self.smooth_btn.setEnabled(
                isinstance(self.label_manager.selected_layer, napari.layers.Labels)
            )
        )

        smoothbox.setLayout(smooth_boxlayout)
        layout = QVBoxLayout()
        layout.addWidget(smoothbox)
        self.setLayout(layout)

    def _smooth_objects(self) -> None:
        """Smooth objects by using a median filter."""

        action = median_filter
        smoothed = process_action_seg(
            self.label_manager.selected_layer.data,
            action,
            basename=self.label_manager.selected_layer.name,
            size=self.median_radius_field.value(),
        )
        if smoothed is not None:
            self.label_manager.selected_layer = self.viewer.add_labels(
                smoothed,
                name=self.label_manager.selected_layer.name + "_medianfilter",
                scale=self.label_manager.selected_layer.scale,
            )
