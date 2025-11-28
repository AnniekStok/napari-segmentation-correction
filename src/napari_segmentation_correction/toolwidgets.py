import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from napari_segmentation_correction.tool_widgets.connected_components import (
    ConnectedComponents,
)
from napari_segmentation_correction.tool_widgets.erosion_dilation_widget import (
    ErosionDilationWidget,
)
from napari_segmentation_correction.tool_widgets.image_calculator import ImageCalculator
from napari_segmentation_correction.tool_widgets.label_boundaries import LabelBoundaries
from napari_segmentation_correction.tool_widgets.label_interpolator import (
    InterpolationWidget,
)
from napari_segmentation_correction.tool_widgets.select_delete_widget import (
    SelectDeleteMask,
)
from napari_segmentation_correction.tool_widgets.smoothing_widget import SmoothingWidget
from napari_segmentation_correction.tool_widgets.threshold_widget import ThresholdWidget


class ToolWidgets(QWidget):
    """Toolbox widgets for editing labels"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self.viewer = viewer

        self.edit_layout = QVBoxLayout()

        ### Add widget for connected component labeling
        conn_comp_widget = ConnectedComponents(self.viewer)
        self.edit_layout.addWidget(conn_comp_widget)

        ### Add widget for label boundaries
        label_boundary_widget = LabelBoundaries(self.viewer)
        self.edit_layout.addWidget(label_boundary_widget)

        ### Add widget for smoothing labels
        smooth_widget = SmoothingWidget(self.viewer)
        self.edit_layout.addWidget(smooth_widget)

        ### Add widget for eroding/dilating labels
        erode_dilate_widget = ErosionDilationWidget(self.viewer)
        self.edit_layout.addWidget(erode_dilate_widget)

        ### Threshold image
        threshold_widget = ThresholdWidget(self.viewer)
        self.edit_layout.addWidget(threshold_widget)

        # Add image calculator
        image_calc = ImageCalculator(self.viewer)
        self.edit_layout.addWidget(image_calc)

        # Add widget for selecting/deleting by mask
        select_del = SelectDeleteMask(self.viewer)
        self.edit_layout.addWidget(select_del)

        # Add widget for interpolating masks
        interpolation_widget = InterpolationWidget(self.viewer)
        self.edit_layout.addWidget(interpolation_widget)

        self.setLayout(self.edit_layout)
