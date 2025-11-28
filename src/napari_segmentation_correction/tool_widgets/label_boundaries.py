import napari
import numpy as np
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
)
from skimage.segmentation import find_boundaries

from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action_seg,
)
from napari_segmentation_correction.tool_widgets.base_tool_widget import BaseToolWidget


def compute_boundaries(seg: np.ndarray) -> np.ndarray:
    """Compute the boundaries and label by segmentation"""

    boundaries = find_boundaries(seg)
    return np.multiply(seg, boundaries)


class LabelBoundaries(BaseToolWidget):
    """Compute the boundaries of a label image"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(napari.layers.Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        box = QGroupBox("Label boundaries")
        box_layout = QVBoxLayout()
        self.compute_btn = QPushButton("Run")
        self.compute_btn.clicked.connect(self._compute_boundaries)
        self.compute_btn.setEnabled(self.layer is not None)
        self.update_status.connect(
            lambda: self.compute_btn.setEnabled(self.layer is not None)
        )

        box_layout.addWidget(self.compute_btn)
        box.setLayout(box_layout)

        layout = QVBoxLayout()
        layout.addWidget(box)
        self.setLayout(layout)

    def _compute_boundaries(self) -> None:
        """Compute the label boundaries of a label image"""

        action = compute_boundaries
        boundaries = process_action_seg(
            self.layer.data,
            action,
            basename=self.layer.name,
        )

        if boundaries is not None:
            self.layer = self.viewer.add_labels(
                boundaries,
                name=self.layer.name + "_label_boundaries",
                scale=self.layer.scale,
            )
