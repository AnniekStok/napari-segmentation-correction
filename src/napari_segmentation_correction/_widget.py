"""
Napari plugin widget for editing N-dimensional label data
"""

import napari
from napari.layers import Labels
from napari_orthogonal_views.ortho_view_manager import _get_manager
from qtpy.QtWidgets import (
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .connected_components import ConnectedComponents
from .erosion_dilation_widget import ErosionDilationWidget
from .image_calculator import ImageCalculator
from .label_interpolator import InterpolationWidget
from .layer_controls import LayerControlsWidget
from .layer_manager import LayerManager
from .plot_widget import PlotWidget
from .regionprops_widget import RegionPropsWidget
from .select_delete_widget import SelectDeleteMask
from .size_filter_widget import SizeFilterWidget
from .smoothing_widget import SmoothingWidget
from .threshold_widget import ThresholdWidget


class AnnotateLabelsND(QWidget):
    """Widget for manual correction of label data, for example to prepare ground truth data for training a segmentation model"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self.source_labels = None
        self.target_labels = None
        self.points = None
        self.copy_points = None
        self.edit_layout = QVBoxLayout()
        self.tab_widget = QTabWidget(self)
        self.option_labels = None

        ### Add label manager
        self.label_manager = LayerManager(self.viewer)

        ### Add layer controls widget
        self.layer_controls = LayerControlsWidget(self.viewer, self.label_manager)

        ### activate orthogonal views and register custom function
        def label_options_click_hook(orig_layer, copied_layer):
            copied_layer.mouse_drag_callbacks.append(
                lambda layer, event: self.layer_controls.copy_label_widget.sync_click(
                    orig_layer, layer, event
                )
            )

        orth_view_manager = _get_manager(self.viewer)
        orth_view_manager.register_layer_hook(Labels, label_options_click_hook)

        ### Add widget for connected component labeling
        conn_comp_widget = ConnectedComponents(self.viewer, self.label_manager)
        self.edit_layout.addWidget(conn_comp_widget)

        ### Add widget for size filtering
        size_filter_widget = SizeFilterWidget(self.viewer, self.label_manager)
        self.edit_layout.addWidget(size_filter_widget)

        ### Add widget for smoothing labels
        smooth_widget = SmoothingWidget(self.viewer, self.label_manager)
        self.edit_layout.addWidget(smooth_widget)

        ### Add widget for eroding/dilating labels
        erode_dilate_widget = ErosionDilationWidget(self.viewer, self.label_manager)
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
        interpolation_widget = InterpolationWidget(self.viewer, self.label_manager)
        self.edit_layout.addWidget(interpolation_widget)

        ### Add layer controls widget to tab
        controls_scroll_area = QScrollArea()
        controls_scroll_area.setWidget(self.layer_controls)
        controls_scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(controls_scroll_area, "Layer Controls")
        self.tab_widget.setCurrentIndex(1)

        ### add combined editing widgets widgets
        self.edit_widgets = QWidget()
        self.edit_widgets.setLayout(self.edit_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.edit_widgets)
        scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_area, "Editing")

        ### Add widget for reginproperties
        self.regionprops_widget = RegionPropsWidget(self.viewer, self.label_manager)
        props_scroll_area = QScrollArea()
        props_scroll_area.setWidget(self.regionprops_widget)
        props_scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(props_scroll_area, "Region properties")

        ### Add widget for displaying plot with regionprops
        self.plot_widget = PlotWidget(self.label_manager)
        self.tab_widget.addTab(self.plot_widget, "Plot")

        # Add the tab widget to the main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tab_widget)
        self.setLayout(self.main_layout)
