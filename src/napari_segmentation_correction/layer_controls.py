import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
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
from napari_segmentation_correction.plane_slider_widget import PlaneSliderWidget


class LayerControlsWidget(QWidget):
    """Widgets for additional layer controls, including setting layer dimensions, plane
    sliders, copy-pasting between layers, and saving layer data in different formats."""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer

        layout = QVBoxLayout()

        ### option to convert dask array to in memory numpy array
        convert_to_numpy_widget = ConvertToNumpyWidget(self.viewer)
        layout.addWidget(convert_to_numpy_widget)

        ### layer dimensions
        self.dimension_widget = DimensionWidget(self.viewer)
        layout.addWidget(self.dimension_widget)

        ### plane sliders
        plane_sliders = PlaneSliderWidget(self.viewer)
        layout.addWidget(plane_sliders)

        ### Add widget for copy-pasting labels from one layer to another
        self.copy_label_widget = CopyLabelWidget(self.viewer)
        layout.addWidget(self.copy_label_widget)

        ### Add widget to save image and label layer data
        save_labels = SaveLabelsWidget(self.viewer)
        layout.addWidget(save_labels)

        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)
