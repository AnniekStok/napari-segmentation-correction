"""
Napari plugin widget for editing N-dimensional label data
"""

import napari
from napari_orthogonal_views.ortho_view_manager import _get_manager
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .connected_components import ConnectedComponents
from .copy_label_widget import CopyLabelWidget
from .erosion_dilation_widget import ErosionDilationWidget
from .image_calculator import ImageCalculator
from .label_interpolator import InterpolationWidget
from .label_option_layer import LabelOptions, sync_click
from .layer_manager import LayerManager
from .point_filter import PointFilter
from .regionprops_widget import RegionPropsWidget
from .save_labels_widget import SaveLabelsWidget
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
        self.outputdir = None
        self.edit_layout = QVBoxLayout()
        self.tab_widget = QTabWidget(self)
        self.option_labels = None

        ### activate orthogonal views and register custom function
        def label_options_click_hook(orig_layer, copied_layer):
            copied_layer.mouse_drag_callbacks.append(
                lambda layer, event: sync_click(orig_layer, layer, event)
            )

        orth_view_manager = _get_manager(self.viewer)
        orth_view_manager.register_layer_hook(LabelOptions, label_options_click_hook)

        ### specify output directory
        outputbox_layout = QHBoxLayout()
        self.outputdirbtn = QPushButton("Select output directory")
        self.output_path = QLineEdit()
        outputbox_layout.addWidget(self.outputdirbtn)
        outputbox_layout.addWidget(self.output_path)
        self.outputdirbtn.clicked.connect(self._on_get_output_dir)
        self.edit_layout.addLayout(outputbox_layout)

        ### create the dropdown for selecting label images
        self.label_manager = LayerManager(self.viewer)
        self.edit_layout.addWidget(self.label_manager)

        ## Add button to clear all layers
        self.clear_btn = QPushButton("Clear all layers")
        self.clear_btn.setEnabled(len(self.viewer.layers) > 0)
        self.viewer.layers.events.removed.connect(
            lambda: self.clear_btn.setEnabled(len(self.viewer.layers) > 0)
        )
        self.viewer.layers.events.inserted.connect(
            lambda: self.clear_btn.setEnabled(len(self.viewer.layers) > 0)
        )

        self.clear_btn.clicked.connect(self._clear_layers)
        self.edit_layout.addWidget(self.clear_btn)

        ### Add widget to save labels
        save_labels = SaveLabelsWidget(self.viewer, self.label_manager)
        self.edit_layout.addWidget(save_labels)

        ### Add widget for filtering by points layer
        point_filter = PointFilter(self.viewer, self.label_manager)
        self.edit_layout.addWidget(point_filter)

        ### Add widget for copy-pasting labels from one layer to another
        copy_label_widget = CopyLabelWidget(self.viewer, self.label_manager)
        self.edit_layout.addWidget(copy_label_widget)

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

        ## add combined editing widgets widgets
        self.edit_widgets = QWidget()
        self.edit_widgets.setLayout(self.edit_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.edit_widgets)
        scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_area, "Editing")
        self.tab_widget.setCurrentIndex(1)

        ### Add widget for adding overview table
        self.regionprops_widget = RegionPropsWidget(self.viewer, self.label_manager)
        props_scroll_area = QScrollArea()
        props_scroll_area.setWidget(self.regionprops_widget)
        props_scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(props_scroll_area, "Region properties")

        # Add the tab widget to the main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tab_widget)
        self.setLayout(self.main_layout)

    def _on_get_output_dir(self) -> None:
        """Show a dialog window to let the user pick the output directory."""

        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path.setText(path)
            self.outputdir = str(self.output_path.text())

    def _clear_layers(self) -> None:
        """Clear all the layers in the viewer"""

        msg = QMessageBox()
        msg.setWindowTitle("Remove all layers?")
        msg.setText("Are you sure you want to remove all layers from the viewer?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if msg.exec_() == QMessageBox.Ok:
            if self.regionprops_widget.table is not None:
                self.regionprops_widget.table.hide()
                self.regionprops_widget.table = None
                self.edit_layout.update()

            self.viewer.layers.clear()
