"""
Toolbox for editing 2D-5D label data.
"""

from napari.layers import Labels
from napari.viewer import Viewer
from napari_orthogonal_views.ortho_view_manager import _get_manager
from qtpy.QtWidgets import (
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from napari_segmentation_correction.layer_controls import LayerControlsWidget
from napari_segmentation_correction.plot_widget import PlotWidget
from napari_segmentation_correction.regionprops_widget import RegionPropsWidget
from napari_segmentation_correction.toolwidgets import ToolWidgets


class LabelToolbox(QWidget):
    """Collection of toolbox widgets that help to correct and analyze segmentation labels."""

    def __init__(self, viewer: Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.source_labels = None
        self.target_labels = None
        self.points = None
        self.copy_points = None
        self.tab_widget = QTabWidget(self)
        self.option_labels = None

        ### Add layer controls widget
        self.layer_controls = LayerControlsWidget(self.viewer)

        ### activate orthogonal views and register custom function
        def label_options_click_hook(orig_layer, copied_layer):
            copied_layer.mouse_drag_callbacks.append(
                lambda layer, event: self.layer_controls.copy_label_widget.sync_click(
                    orig_layer, layer, event
                )
            )

        self.orth_view_manager = _get_manager(self.viewer)
        self.orth_view_manager.register_layer_hook(Labels, label_options_click_hook)

        ### Add layer controls widget to tab
        controls_scroll_area = QScrollArea()
        controls_scroll_area.setWidget(self.layer_controls)
        controls_scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(controls_scroll_area, "Extra Layer Controls")
        self.tab_widget.setCurrentIndex(1)

        ### add combined tool widgets
        self.edit_widgets = ToolWidgets(self.viewer)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.edit_widgets)
        scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_area, "Tools")

        ### Add widget for regionproperties
        self.regionprops_widget = RegionPropsWidget(self.viewer)
        props_scroll_area = QScrollArea()
        props_scroll_area.setWidget(self.regionprops_widget)
        props_scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(props_scroll_area, "Region properties")

        # connect dimension widget signal to regionprops widget
        self.layer_controls.dimension_widget.update_status.connect(
            self.regionprops_widget.update_properties_and_callback
        )  # forward signal

        ### Add widget for displaying plot with regionprops
        self.plot_widget = PlotWidget(self.viewer)
        self.tab_widget.addTab(self.plot_widget, "Plot")

        # Add the tab widget to the main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tab_widget)
        self.setLayout(self.main_layout)

        self.setMinimumWidth(400)

    def deleteLater(self):
        """Ensure ortho views get cleaned up properly"""

        self.orth_view_manager.cleanup()
        super().deleteLater()
