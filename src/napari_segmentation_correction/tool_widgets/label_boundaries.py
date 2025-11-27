import napari
import numpy as np
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.segmentation import find_boundaries

from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action_seg,
)
from napari_segmentation_correction.layer_control_widgets.layer_manager import (
    LayerManager,
)


def compute_boundaries(seg: np.ndarray) -> np.ndarray:
    """Compute the boundaries and label by segmentation"""

    boundaries = find_boundaries(seg)
    return np.multiply(seg, boundaries)


class LabelBoundaries(QWidget):
    """Compute the boundaries of a label image"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager

        boundary_box = QGroupBox("Label boundaries")
        boundary_box_layout = QVBoxLayout()
        self.compute_btn = QPushButton("Run")
        self.compute_btn.clicked.connect(self._compute_boundaries)
        self.compute_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        boundary_box_layout.addWidget(self.compute_btn)
        boundary_box.setLayout(boundary_box_layout)

        self.label_manager.layer_update.connect(self._update_button_state)

        main_layout = QVBoxLayout()
        main_layout.addWidget(boundary_box)
        self.setLayout(main_layout)

    def _update_button_state(self) -> None:
        """Update button state"""

        self.compute_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )

    def _compute_boundaries(self) -> None:
        """Compute the label boundaries of a label image"""

        action = compute_boundaries
        boundaries = process_action_seg(
            self.label_manager.selected_layer.data,
            action,
            basename=self.label_manager.selected_layer.name,
        )

        if boundaries is not None:
            self.label_manager.selected_layer = self.viewer.add_labels(
                boundaries,
                name=self.label_manager.selected_layer.name + "_label_boundaries",
                scale=self.label_manager.selected_layer.scale,
            )
