"""
Napari plugin widget for editing N-dimensional label data
"""

import os
import shutil

import dask.array as da
import napari
import numpy as np
import tifffile
from napari_plane_sliders._plane_slider_widget import PlaneSliderWidget
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage import measure
from skimage.io import imread

from ._custom_table_widget import ColoredTableWidget
from .erosion_dilation_widget import ErosionDilationWidget
from .image_calculator import ImageCalculator
from .layer_manager import LayerManager
from .point_filter import PointFilter
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
        self.table = None
        self.points = None
        self.copy_points = None
        self.outputdir = None
        self.settings_layout = QVBoxLayout()
        self.tab_widget = QTabWidget(self)
        self.option_labels = None

        ### specify output directory
        outputbox_layout = QHBoxLayout()
        self.outputdirbtn = QPushButton("Select output directory")
        self.output_path = QLineEdit()
        outputbox_layout.addWidget(self.outputdirbtn)
        outputbox_layout.addWidget(self.output_path)
        self.outputdirbtn.clicked.connect(self._on_get_output_dir)
        self.settings_layout.addLayout(outputbox_layout)

        ### create the dropdown for selecting label images
        self.label_manager = LayerManager(self.viewer)
        self.settings_layout.addWidget(self.label_manager)

        ### Add option to convert dask array to in-memory array
        self.convert_to_array_btn = QPushButton("Convert to in-memory array")
        self.convert_to_array_btn.setEnabled(
            self.label_manager.selected_layer is not None and isinstance(self.label_manager.selected_layer.data, da.core.Array)
        )
        self.convert_to_array_btn.clicked.connect(self._convert_to_array)
        self.settings_layout.addWidget(self.convert_to_array_btn)

        ### Add widget for adding overview table
        self.table_btn = QPushButton("Show table")
        self.table_btn.clicked.connect(self._create_summary_table)
        self.table_btn.clicked.connect(
            lambda: self.tab_widget.setCurrentIndex(0)
        )
        if self.label_manager.selected_layer is not None:
            self.table_btn.setEnabled(True)
        self.settings_layout.addWidget(self.table_btn)

        ## Add save labels widget
        self.save_btn = QPushButton("Save labels")
        self.save_btn.clicked.connect(self._save_labels)
        self.settings_layout.addWidget(self.save_btn)

        ## Add button to clear all layers
        self.clear_btn = QPushButton("Clear all layers")
        self.clear_btn.clicked.connect(self._clear_layers)
        self.settings_layout.addWidget(self.clear_btn)

        ### Add widget for filtering by points layer
        point_filter = PointFilter(self.viewer, self.label_manager)
        self.settings_layout.addWidget(point_filter)

        ### Add widget for copy-pasting labels from one layer to another
        copy_labels_box = QGroupBox("Copy-paste labels")
        copy_labels_layout = QVBoxLayout()

        add_option_layer_btn = QPushButton(
            "Add layer with different label options from folder"
        )
        add_option_layer_btn.clicked.connect(self._add_option_layer)
        convert_to_option_layer_btn = QPushButton(
            "Convert current labels layer to label options layer"
        )
        convert_to_option_layer_btn.clicked.connect(
            self._convert_to_option_layer
        )

        copy_labels_layout.addWidget(add_option_layer_btn)
        copy_labels_layout.addWidget(convert_to_option_layer_btn)

        copy_labels_box.setLayout(copy_labels_layout)
        self.settings_layout.addWidget(copy_labels_box)

        ### Add widget for size filtering
        self.settings_layout.addWidget(SizeFilterWidget(self.viewer, self.label_manager))

        self.setLayout(self.settings_layout)

        ### Add widget for smoothing labels
        smooth_widget = SmoothingWidget(self.viewer, self.label_manager)
        self.settings_layout.addWidget(smooth_widget)

        ### Add widget for eroding/dilating labels
        erode_dilate_widget = ErosionDilationWidget(self.viewer, self.label_manager)
        self.settings_layout.addWidget(erode_dilate_widget)

        ### Threshold image
        threshold_widget = ThresholdWidget(self.viewer)
        self.settings_layout.addWidget(threshold_widget)

        # Add image calculator
        image_calc = ImageCalculator(self.viewer)
        self.settings_layout.addWidget(image_calc)

        ## add plane viewing widget
        self.slider_table_widget = QWidget()
        self.plane_slider_table_layout = QVBoxLayout()
        self.plane_slider_table_layout.addWidget(
            PlaneSliderWidget(self.viewer)
        )
        self.slider_table_widget.setLayout(self.plane_slider_table_layout)
        self.tab_widget.addTab(self.slider_table_widget, "Plane Viewing")

        ## add combined settings widgets
        self.settings_widgets = QWidget()
        self.settings_widgets.setLayout(self.settings_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.settings_widgets)
        scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_area, "Settings")

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


    def _update_source_labels(self, selected_layer) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.source_labels = None
        else:
            self.source_labels = self.viewer.layers[selected_layer]
            self.source_label_dropdown.setCurrentText(selected_layer)

    def _update_target_labels(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'target labels' layer for copying labels to."""

        if selected_layer == "":
            self.target_labels = None
        else:
            self.target_labels = self.viewer.layers[selected_layer]
            self.target_label_dropdown.setCurrentText(selected_layer)



    def _convert_to_array(self) -> None:
        """Convert from dask array to in-memory array. This is necessary for manual editing using the label tools (brush, eraser, fill bucket)."""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            stack = []
            for i in range(self.label_manager.selected_layer.data.shape[0]):
                current_stack = self.label_manager.selected_layer.data[i].compute()
                stack.append(current_stack)
            self.label_manager.selected_layer.data = np.stack(stack, axis=0)

    def _create_summary_table(self) -> None:
        """Create table displaying the sizes of the different labels in the current stack"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            tp = self.viewer.dims.current_step[0]
            current_stack = self.label_manager.selected_layer.data[
                tp
            ].compute()  # Compute the current stack
            props = measure.regionprops_table(
                current_stack, properties=["label", "area", "centroid"]
            )
            if hasattr(self.label_manager.selected_layer, "properties"):
                self.label_manager.selected_layer.properties = props
            if hasattr(self.label_manager.selected_layer, "features"):
                self.label_manager.selected_layer.features = props

        else:
            if len(self.label_manager.selected_layer.data.shape) == 4:
                tp = self.viewer.dims.current_step[0]
                props = measure.regionprops_table(
                    self.label_manager.selected_layer.data[tp],
                    properties=["label", "area", "centroid"],
                )
                if hasattr(self.label_manager.selected_layer, "properties"):
                    self.label_manager.selected_layer.properties = props
                if hasattr(self.label_manager.selected_layer, "features"):
                    self.label_manager.selected_layer.features = props

            elif len(self.label_manager.selected_layer.data.shape) == 3:
                props = measure.regionprops_table(
                    self.label_manager.selected_layer.data, properties=["label", "area", "centroid"]
                )
                if hasattr(self.label_manager.selected_layer, "properties"):
                    self.label_manager.selected_layer.properties = props
                if hasattr(self.label_manager.selected_layer, "features"):
                    self.label_manager.selected_layer.features = props
            else:
                print("input should be a 3D or 4D array")
                self.table = None

        # add the napari-skimage-regionprops inspired table to the viewer
        if self.table is not None:
            self.table.hide()

        if self.viewer is not None:
            self.table = ColoredTableWidget(self.label_manager.selected_layer, self.viewer)
            self.table._set_label_colors_to_rows()
            self.table.setMinimumWidth(500)
            self.plane_slider_table_layout.addWidget(self.table)

    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):

            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False

            else:
                outputdir = os.path.join(
                    self.outputdir, (self.label_manager.selected_layer.name + "_finalresult")
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
                        np.array(current_stack, dtype="uint16"),
                    )
                return True

        elif len(self.label_manager.selected_layer.data.shape) == 4:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )
            for i in range(self.label_manager.selected_layer.data.shape[0]):
                labels_data = self.label_manager.selected_layer.data[i].astype(np.uint16)
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
                labels_data = self.label_manager.selected_layer.data.astype(np.uint16)
                tifffile.imwrite(filename, labels_data)

        else:
            print("labels should be a 3D or 4D array")

    def _clear_layers(self) -> None:
        """Clear all the layers in the viewer"""

        if self.table is not None:
            self.table.hide()
            self.table = None
            self.settings_layout.update()

        self.viewer.layers.clear()

    def _add_option_layer(self):
        """Add a new labels layer that contains different alternative segmentations as channels, and add a function to select and copy these cells through shift-clicking"""

        path = QFileDialog.getExistingDirectory(
            self, "Select Label Image Parent Folder"
        )
        if path:
            label_dirs = sorted(
                [
                    d
                    for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))
                ]
            )
            label_stacks = []
            for d in label_dirs:
                # n dirs indicates number of channels
                label_files = sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(path, d))
                        if ".tif" in f
                    ]
                )
                label_imgs = []
                for f in label_files:
                    # n label_files indicates n time points
                    img = imread(os.path.join(path, d, f))
                    label_imgs.append(img)

                if len(label_imgs) > 1:
                    label_stack = np.stack(label_imgs, axis=0)
                    label_stacks.append(label_stack)
                else:
                    label_stacks.append(img)

            if len(label_stacks) > 1:
                self.option_labels = np.stack(label_stacks, axis=0)
            elif len(label_stacks) == 1:
                self.option_labels = label_stacks[0]

            n_channels = len(label_dirs)
            n_timepoints = len(label_files)
            if len(img.shape) == 3:
                n_slices = img.shape[0]
            elif len(img.shape) == 2:
                n_slices = 1

            self.option_labels = self.option_labels.reshape(
                n_channels,
                n_timepoints,
                n_slices,
                img.shape[-2],
                img.shape[-1],
            )
            self.option_labels = self.viewer.add_labels(
                self.option_labels, name="label options"
            )

        viewer = self.viewer

        @viewer.mouse_drag_callbacks.append
        def cell_copied(viewer, event):
            if (
                event.type == "mouse_press"
                and "Shift" in event.modifiers
                and viewer.layers.selection.active == self.option_labels
            ):
                coords = self.option_labels.world_to_data(event.position)
                coords = [int(c) for c in coords]
                selected_label = self.option_labels.get_value(coords)
                mask = (
                    self.option_labels.data[coords[0], coords[1], :, :, :]
                    == selected_label
                )

                if isinstance(self.label_manager.selected_layer.data, da.core.Array):
                    target_stack = self.label_manager.selected_layer.data[coords[-4]].compute()
                    orig_label = target_stack[
                        coords[-3], coords[-2], coords[-1]
                    ]
                    if orig_label != 0:
                        target_stack[target_stack == orig_label] = 0
                    target_stack[mask] = np.max(target_stack) + 1
                    self.label_manager.selected_layer.data[coords[-4]] = target_stack
                    self.label_manager.selected_layer.data = self.label_manager.selected_layer.data

                else:
                    if len(self.label_manager.selected_layer.data.shape) == 3:
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.label_manager.selected_layer.data[
                                self.label_manager.selected_layer.data == orig_label
                            ] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[mask] = np.max(self.label_manager.selected_layer.data) + 1
                        self.label_manager.selected_layer.data = self.label_manager.selected_layer.data

                    elif len(self.label_manager.selected_layer.data.shape) == 4:
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-4], coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.label_manager.selected_layer.data[coords[-4]][
                                self.label_manager.selected_layer.data[coords[-4]] == orig_label
                            ] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[coords[-4]][mask] = (
                            np.max(self.label_manager.selected_layer.data) + 1
                        )
                        self.label_manager.selected_layer.data = self.label_manager.selected_layer.data

                    elif len(self.label_manager.selected_layer.data.shape) == 5:
                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setText(
                            "Copy-pasting in 5 dimensions is not implemented, do you want to convert the labels layer to 5 dimensions (tzyx)?"
                        )
                        msg_box.setWindowTitle("Convert to 4 dimensions?")

                        yes_button = msg_box.addButton(QMessageBox.Yes)
                        no_button = msg_box.addButton(QMessageBox.No)

                        msg_box.exec_()

                        if msg_box.clickedButton() == yes_button:
                            self.label_manager.selected_layer.data = self.label_manager.selected_layer.data[0]
                        elif msg_box.clickedButton() == no_button:
                            return False
                    else:
                        print(
                            "copy-pasting in more than 5 dimensions is not supported"
                        )

    def _convert_to_option_layer(self) -> None:

        if len(self.label_manager.selected_layer.data.shape) == 3:
            self.option_labels = self.viewer.add_labels(
                self.label_manager.selected_layer.data.reshape(
                    (
                        1,
                        1,
                    )
                    + self.label_manager.selected_layer.data.shape
                ),
                name="label options",
            )
        elif len(self.label_manager.selected_layer.data.shape) == 4:
            self.option_labels = self.viewer.add_labels(
                self.label_manager.selected_layer.data.reshape((1,) + self.label_manager.selected_layer.data.shape),
                name="label options",
            )
        elif len(self.label_manager.selected_layer.data.shape) == 5:
            self.option_labels = self.viewer.add_labels(
                self.label_manager.selected_layer.data, name="label options"
            )
        else:
            print("labels data must have at least 3 dimensions")
            return

        viewer = self.viewer

        @viewer.mouse_drag_callbacks.append
        def cell_copied(viewer, event):
            if (
                event.type == "mouse_press"
                and "Shift" in event.modifiers
                and viewer.layers.selection.active == self.option_labels
            ):
                coords = self.option_labels.world_to_data(event.position)
                coords = [int(c) for c in coords]
                selected_label = self.option_labels.get_value(coords)
                mask = (
                    self.option_labels.data[coords[0], coords[1], :, :, :]
                    == selected_label
                )

                if isinstance(self.label_manager.selected_layer.data, da.core.Array):
                    target_stack = self.label_manager.selected_layer.data[coords[-4]].compute()
                    orig_label = target_stack[
                        coords[-3], coords[-2], coords[-1]
                    ]
                    if orig_label != 0:
                        target_stack[target_stack == orig_label] = 0
                    target_stack[mask] = np.max(target_stack) + 1
                    self.label_manager.selected_layer.data[coords[-4]] = target_stack
                    self.label_manager.selected_layer.data = self.label_manager.selected_layer.data

                else:
                    if len(self.label_manager.selected_layer.data.shape) == 3:
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.label_manager.selected_layer.data[
                                self.label_manager.selected_layer.data == orig_label
                            ] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[mask] = np.max(self.label_manager.selected_layer.data) + 1
                        self.label_manager.selected_layer.data = self.label_manager.selected_layer.data

                    elif len(self.label_manager.selected_layer.data.shape) == 4:
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-4], coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.label_manager.selected_layer.data[coords[-4]][
                                self.label_manager.selected_layer.data[coords[-4]] == orig_label
                            ] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[coords[-4]][mask] = (
                            np.max(self.label_manager.selected_layer.data) + 1
                        )
                        self.label_manager.selected_layer.data = self.label_manager.selected_layer.data

                    elif len(self.label_manager.selected_layer.data.shape) == 5:
                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setText(
                            "Copy-pasting in 5 dimensions is not implemented, do you want to convert the labels layer to 5 dimensions (tzyx)?"
                        )
                        msg_box.setWindowTitle("Convert to 4 dimensions?")

                        yes_button = msg_box.addButton(QMessageBox.Yes)
                        no_button = msg_box.addButton(QMessageBox.No)

                        msg_box.exec_()

                        if msg_box.clickedButton() == yes_button:
                            self.label_manager.selected_layer.data = self.label_manager.selected_layer.data[0]
                        elif msg_box.clickedButton() == no_button:
                            return False
                    else:
                        print(
                            "copy-pasting in more than 5 dimensions is not supported"
                        )







