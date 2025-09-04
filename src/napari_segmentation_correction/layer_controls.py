import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from .copy_label_widget import CopyLabelWidget
from .layer_manager import LayerManager
from .save_labels_widget import SaveLabelsWidget


class LayerControlsWidget(QWidget):
    """Widget showing region props as a table and plot widget"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager

        layout = QVBoxLayout()

        ### create the dropdown for selecting label images
        layout.addWidget(self.label_manager)

        ### Add widget for copy-pasting labels from one layer to another
        self.copy_label_widget = CopyLabelWidget(self.viewer)
        layout.addWidget(self.copy_label_widget)

        ### Add widget to save labels
        save_labels = SaveLabelsWidget(self.viewer, self.label_manager)
        layout.addWidget(save_labels)

        self.setLayout(layout)
