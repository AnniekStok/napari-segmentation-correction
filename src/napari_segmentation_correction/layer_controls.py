import napari
from napari_plane_sliders import PlaneSliderWidget
from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from napari_segmentation_correction.layer_control_widgets.convert_to_numpy import (
    ConvertToNumpyWidget,
)
from napari_segmentation_correction.layer_control_widgets.copy_label_widget import (
    CopyLabelWidget,
)
from napari_segmentation_correction.layer_control_widgets.dimension_widget import (
    DimensionWidget,
)
from napari_segmentation_correction.layer_control_widgets.save_labels_widget import (
    SaveLabelsWidget,
)


class LayerControlsWidget(QWidget):
    """Widgets for additional layer controls, including setting layer dimensions, plane
    sliders, copy-pasting between layers, and saving layer data in different formats."""

    update_dims = Signal()

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer

        layout = QVBoxLayout()

        ### option to convert dask array to in memory numpy array
        self.convert_to_numpy_widget = ConvertToNumpyWidget(self.viewer)
        layout.addWidget(self.convert_to_numpy_widget)

        ### layer dimensions
        self.dimension_widget = DimensionWidget(self.viewer)
        self.dimension_widget.update_dims.connect(self.update_dims)  # forward signal
        layout.addWidget(self.dimension_widget)

        ### plane sliders
        plane_slider_box = QGroupBox("Plane Sliders")
        plane_slider_layout = QVBoxLayout()
        plane_sliders = PlaneSliderWidget(self.viewer)
        plane_slider_layout.addWidget(plane_sliders)
        plane_slider_box.setLayout(plane_slider_layout)
        layout.addWidget(plane_slider_box)

        ### Add widget for copy-pasting labels from one layer to another
        self.copy_label_widget = CopyLabelWidget(self.viewer)
        layout.addWidget(self.copy_label_widget)

        ### Add widget to save image and label layer data
        save_labels = SaveLabelsWidget(self.viewer)
        layout.addWidget(save_labels)

        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)
