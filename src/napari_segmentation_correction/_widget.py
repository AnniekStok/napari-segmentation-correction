"""
Napari plugin widget for editing N-dimensional label data
"""

import os
import shutil

import dask.array as da
import napari
import numpy as np
import tifffile
from napari.layers import Labels
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
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
from .layer_manager import LayerManager
from .point_filter import PointFilter
from .regionprops_widget import RegionPropsWidget
from .select_delete_widget import SelectDeleteMask
from .size_filter_widget import SizeFilterWidget
from .smoothing_widget import SmoothingWidget
from .threshold_widget import ThresholdWidget
from .view3D import View3D


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

        ### connect event to switch ndisplay because the otherwise the orthogonal views might be frozen (bug)
        self.viewer.dims.events.ndisplay.connect(self.update_3D_tab)

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

        ### Add widget for adding overview table
        self.table_btn = QPushButton("Show table")
        self.regionprops_widget = RegionPropsWidget(self.viewer, self.label_manager)
        self.table_btn.clicked.connect(self.regionprops_widget._create_summary_table)
        self.table_btn.clicked.connect(
            lambda: self.tab_widget.setCurrentIndex(2)
        )
        self.table_btn.setEnabled(isinstance(self.label_manager.selected_layer, Labels))
        self.label_manager.layer_update.connect(
            lambda: self.table_btn.setEnabled(isinstance(self.label_manager.selected_layer, napari.layers.Labels))
        )
        self.edit_layout.addWidget(self.table_btn)

        ## Add save labels widget
        self.save_btn = QPushButton("Save labels")
        self.save_btn.clicked.connect(self._save_labels)
        self.save_btn.setEnabled(isinstance(self.label_manager.selected_layer, Labels))
        self.label_manager.layer_update.connect(
            lambda: self.save_btn.setEnabled(isinstance(self.label_manager.selected_layer, napari.layers.Labels))
        )

        self.edit_layout.addWidget(self.save_btn)

        ## Add button to clear all layers
        self.clear_btn = QPushButton("Clear all layers")
        self.clear_btn.clicked.connect(self._clear_layers)
        self.edit_layout.addWidget(self.clear_btn)

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
        erode_dilate_widget = ErosionDilationWidget(
            self.viewer, self.label_manager
        )
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

        ## add 3d viewing widget
        self.view3d_widget = View3D(self.viewer)
        self.tab_widget.addTab(self.view3d_widget, "3D Viewing")

        ## add combined editing widgets widgets
        self.edit_widgets = QWidget()
        self.edit_widgets.setLayout(self.edit_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.edit_widgets)
        scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_area, "Editing")
        self.tab_widget.setCurrentIndex(1)

        ## add widget for viewing data
        self.tab_widget.addTab(self.regionprops_widget, "Region properties")

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

    def update_3D_tab(self):
        """Silly workaround to ensure that the orthongal views are updated when switching ndims displayed"""
        if self.tab_widget.currentIndex() == 0:
            self.tab_widget.setCurrentIndex(1)
            self.tab_widget.setCurrentIndex(0)

    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):

            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_finalresult"),
            )
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
            os.mkdir(outputdir)

            for i in range(
                self.label_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                current_stack = self.label_manager.selected_layer.data[
                    i
                ].compute()  # Compute the current stack
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(current_stack),
                )

        elif len(self.label_manager.selected_layer.data.shape) == 4:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )
            for i in range(self.label_manager.selected_layer.data.shape[0]):
                labels_data = self.label_manager.selected_layer.data[i].astype(
                    np.uint16
                )
                tifffile.imwrite(
                    (
                        filename.split(".tif")[0]
                        + "_TP"
                        + str(i).zfill(4)
                        + ".tif"
                    ),
                    labels_data,
                )

        elif len(self.label_manager.selected_layer.data.shape) == 3:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )

            if filename:
                labels_data = self.label_manager.selected_layer.data.astype(
                    np.uint16
                )
                tifffile.imwrite(filename, labels_data)

        else:
            print("labels should be a 3D or 4D array")

    def _clear_layers(self) -> None:
        """Clear all the layers in the viewer"""

        if self.regionprops_widget.table is not None:
            self.regionprops_widget.table.hide()
            self.regionprops_widget.table = None
            self.edit_layout.update()

        self.viewer.layers.clear()
