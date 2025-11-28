import napari
import numpy as np
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from napari_segmentation_correction.helpers.base_tool_widget import BaseToolWidget
from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action_seg,
)


def threshold(
    img: np.ndarray, min_val: int | float, max_val: int | float
) -> np.ndarray:
    """Threshold the input image"""

    return (img >= min_val) & (img <= max_val)


class ThresholdWidget(BaseToolWidget):
    """Widget that applies a threshold to an image or labels layer"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(Image, Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        threshold_box = QGroupBox("Threshold")
        threshold_box_layout = QVBoxLayout()

        min_threshold_layout = QHBoxLayout()
        min_threshold_layout.addWidget(QLabel("Min value"))
        self.min_threshold = QDoubleSpinBox()
        self.min_threshold.setMaximum(65535)
        min_threshold_layout.addWidget(self.min_threshold)

        max_threshold_layout = QHBoxLayout()
        max_threshold_layout.addWidget(QLabel("Max value"))
        self.max_threshold = QDoubleSpinBox()
        self.max_threshold.setMaximum(65535)
        self.max_threshold.setValue(65535)
        max_threshold_layout.addWidget(self.max_threshold)

        threshold_box_layout.addLayout(min_threshold_layout)
        threshold_box_layout.addLayout(max_threshold_layout)
        threshold_btn = QPushButton("Run")
        threshold_btn.clicked.connect(self._threshold)

        threshold_btn.setEnabled(self.layer is not None)
        self.update_status.connect(
            lambda: threshold_btn.setEnabled(self.layer is not None)
        )

        threshold_box_layout.addWidget(threshold_btn)
        threshold_box.setLayout(threshold_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(threshold_box)
        self.setLayout(layout)

    def _threshold(self) -> None:
        """Threshold the selected label or intensity image"""

        action = threshold
        mask = process_action_seg(
            seg=self.layer.data,
            action=action,
            basename=self.layer.name,
            min_val=self.min_threshold.value(),
            max_val=self.max_threshold.value(),
        )
        if mask is not None:
            self.viewer.add_labels(
                mask,
                name=self.layer.name + "_threshold",
                scale=self.layer.scale,
            )
