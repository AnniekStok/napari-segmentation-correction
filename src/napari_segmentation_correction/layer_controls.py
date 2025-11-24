import napari
from napari_plane_sliders import PlaneSliderWidget
from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from .copy_label_widget import CopyLabelWidget
from .dimension_widget import DimensionWidget
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

        ### layer dimensions
        self.dimension_widget = DimensionWidget(self.viewer)
        self.dimension_widget.dims_updated.connect(self._update_dims)
        layout.addWidget(self.dimension_widget)

        ### plane sliders
        plane_slider_box = QGroupBox("Plane Sliders")
        plane_slider_layout = QVBoxLayout()
        self.plane_sliders = PlaneSliderWidget(self.viewer)
        plane_slider_layout.addWidget(self.plane_sliders)
        plane_slider_box.setLayout(plane_slider_layout)
        layout.addWidget(plane_slider_box)

        ### Add widget for copy-pasting labels from one layer to another
        self.copy_label_widget = CopyLabelWidget(self.viewer)
        layout.addWidget(self.copy_label_widget)

        ### Add widget to save labels
        save_labels = SaveLabelsWidget(self.viewer, self.label_manager)
        layout.addWidget(save_labels)

        self.setLayout(layout)

    def _update_dims(self) -> None:
        """If the current layer is the selected labels layer, emit update signal to notify
        the regionprops widget to update its properties."""

        if self.dimension_widget.layer == self.label_manager.selected_layer:
            self.label_manager.layer_update.emit()
