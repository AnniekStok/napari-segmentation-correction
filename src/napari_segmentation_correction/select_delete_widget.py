import copy
import functools
import os
import shutil

import dask.array as da
import napari
import numpy as np
import tifffile
from napari.layers import Labels
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread

from .layer_dropdown import LayerDropdown


class SelectDeleteMask(QWidget):
    """Widget to select labels to keep or to delete based on overlap with a mask."""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer
        self.image1_layer = None
        self.mask_layer = None
        self.outputdir = None

        ### Add one image to another
        select_delete_box = QGroupBox("Select / Delete labels by mask")
        select_delete_box_layout = QVBoxLayout()

        image1_layout = QHBoxLayout()
        image1_layout.addWidget(QLabel("Labels"))
        self.image1_dropdown = LayerDropdown(self.viewer, (Labels))
        self.image1_dropdown.layer_changed.connect(self._update_image1)
        image1_layout.addWidget(self.image1_dropdown)

        image2_layout = QHBoxLayout()
        image2_layout.addWidget(QLabel("Mask"))
        self.mask_dropdown = LayerDropdown(self.viewer, (Labels))
        self.mask_dropdown.layer_changed.connect(self._update_image2)
        image2_layout.addWidget(self.mask_dropdown)

        select_delete_box_layout.addLayout(image1_layout)
        select_delete_box_layout.addLayout(image2_layout)

        self.stack_checkbox = QCheckBox("Apply 3D mask to all time points in 4D array")
        self.stack_checkbox.setEnabled(False)
        select_delete_box_layout.addWidget(self.stack_checkbox)

        self.select_btn = QPushButton("Select labels")
        self.select_btn.clicked.connect(self.select_labels)
        select_delete_box_layout.addWidget(self.select_btn)

        self.delete_btn = QPushButton("Delete labels")
        self.delete_btn.clicked.connect(self.delete_labels)
        select_delete_box_layout.addWidget(self.delete_btn)

        select_delete_box.setLayout(select_delete_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(select_delete_box)
        self.setLayout(main_layout)

    def _update_image1(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

        # update the checkbox and buttons as needed needed
        if self.mask_layer is not None and self.image1_layer is not None:
            self.select_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            if len(self.image1_layer.data.shape) == len(self.mask_layer.data.shape) + 1:
                self.stack_checkbox.setEnabled(True)
            else:
                self.stack_checkbox.setEnabled(False)
                self.stack_checkbox.setCheckState(False)
        else:
            self.select_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.stack_checkbox.setEnabled(False)
            self.stack_checkbox.setCheckState(False)

    def _update_image2(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.mask_layer = None
        else:
            self.mask_layer = self.viewer.layers[selected_layer]
            self.mask_dropdown.setCurrentText(selected_layer)

        # update the checkbox and buttons as needed
        if self.mask_layer is not None and self.image1_layer is not None:
            self.select_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            if len(self.image1_layer.data.shape) == len(self.mask_layer.data.shape) + 1:
                self.stack_checkbox.setEnabled(True)
            else:
                self.stack_checkbox.setEnabled(False)
                self.stack_checkbox.setCheckState(False)
        else:
            self.select_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.stack_checkbox.setEnabled(False)
            self.stack_checkbox.setCheckState(False)

    def select_labels(self):

        # check data dimensions first
        image_shape = self.image1_layer.data.shape
        mask_shape = self.mask_layer.data.shape

        if len(image_shape) == len(mask_shape) + 1 and image_shape[1:] == mask_shape:
            # apply mask to single time point or to full stack depending on checkbox state
            if self.stack_checkbox.isChecked():
                # loop over all time points
                print('applying the mask to all time points')
                # check if the data is a dask array
                if isinstance(self.image1_layer.data, da.core.Array):
                    if self.outputdir is None:
                        self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

                    outputdir = os.path.join(
                        self.outputdir,
                        (self.image1_layer.name + "_filtered_labels"),
                    )
                    if os.path.exists(outputdir):
                        shutil.rmtree(outputdir)
                    os.mkdir(outputdir)

                    for i in range(
                        self.image1_layer.data.shape[0]
                    ):  # Loop over the first dimension
                        current_stack = self.image1_layer.data[
                            i
                        ].compute()  # Compute the current stack

                        to_keep = np.unique(current_stack[self.mask_layer.data > 0])
                        filtered_mask = functools.reduce(np.logical_or, (current_stack == val for val in to_keep))
                        filtered_data = np.where(filtered_mask, current_stack, 0)

                        tifffile.imwrite(
                            os.path.join(
                                outputdir,
                                (
                                    self.image1_layer.name
                                    + "_filtered_labels_TP"
                                    + str(i).zfill(4)
                                    + ".tif"
                                ),
                            ),
                            np.array(filtered_data, dtype="uint16"),
                        )

                    file_list = [
                        os.path.join(outputdir, fname)
                        for fname in os.listdir(outputdir)
                        if fname.endswith(".tif")
                    ]
                    self.image1_layer = self.viewer.add_labels(
                        da.stack([imread(fname) for fname in sorted(file_list)]),
                        name=self.image1_layer.name + "_filtered_labels",
                    )

                else:
                    for tp in range(self.image1_layer.data.shape[0]):
                        to_keep = np.unique(self.image1_layer.data[tp][self.mask_layer.data > 0])
                        filtered_mask = functools.reduce(np.logical_or, (self.image1_layer.data[tp] == val for val in to_keep))
                        filtered_data_tp = np.where(filtered_mask, self.image1_layer.data[tp], 0)
                        self.image1_layer.data[tp] = filtered_data_tp

            else:
                tp = self.viewer.dims.current_step[0]
                print('applying the mask to the current time point only', tp)
                if isinstance(self.image1_layer.data, da.core.Array):
                    outputdir = QFileDialog.getExistingDirectory(self, "Please select the directory that holds the images. Data will be changed here. Selecting a new empty directory will create a copy of all data")

                    if len(os.listdir(outputdir)) == 0:
                        for i in range(
                            self.image1_layer.data.shape[0]
                        ):  # Loop over the first dimension
                            current_stack = self.image1_layer.data[
                                i
                            ].compute()  # Compute the current stack

                            if i == tp:
                                to_keep = np.unique(current_stack[self.mask_layer.data > 0])
                                filtered_mask = functools.reduce(np.logical_or, (current_stack == val for val in to_keep))
                                current_stack = np.where(filtered_mask, current_stack, 0)
                            tifffile.imwrite(
                                os.path.join(
                                    outputdir,
                                    (
                                        self.image1_layer.name
                                        + "_filtered_labels_TP"
                                        + str(i).zfill(4)
                                        + ".tif"
                                    ),
                                ),
                                np.array(current_stack, dtype="uint16"),
                            )

                            file_list = sorted([
                                os.path.join(outputdir, fname)
                                for fname in os.listdir(outputdir)
                                if fname.endswith(".tif")
                            ])
                    else:
                        current_stack = self.image1_layer.data[
                            tp
                        ].compute()  # Compute the current stack

                        to_keep = np.unique(current_stack[self.mask_layer.data > 0])
                        filtered_mask = functools.reduce(np.logical_or, (current_stack == val for val in to_keep))
                        current_stack = np.where(filtered_mask, current_stack, 0)

                        file_list = sorted([
                            os.path.join(outputdir, fname)
                            for fname in os.listdir(outputdir)
                            if fname.endswith(".tif")
                        ])

                        tifffile.imwrite(
                            file_list[tp],
                            np.array(current_stack, dtype="uint16"),
                        )

                    self.image1_layer = self.viewer.add_labels(
                        da.stack([imread(fname) for fname in file_list]),
                        name=self.image1_layer.name + "_filtered_labels",
                    )

                else:
                    tp = self.viewer.dims.current_step[0]
                    to_keep = np.unique(self.image1_layer.data[tp][self.mask_layer.data > 0])
                    filtered_mask = functools.reduce(np.logical_or, (self.image1_layer.data[tp] == val for val in to_keep))
                    filtered_data_tp = np.where(filtered_mask, self.image1_layer.data[tp], 0)
                    self.image1_layer.data[tp] = filtered_data_tp

        elif image_shape == mask_shape:
            if isinstance(self.image1_layer.data, da.core.Array):
                if self.outputdir is None:
                    self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

                outputdir = os.path.join(
                    self.outputdir,
                    (self.image1_layer.name + "_filtered_labels"),
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(
                    self.image1_layer.data.shape[0]
                ):  # Loop over the first dimension
                    current_stack = self.image1_layer.data[
                        i
                    ].compute()  # Compute the current stack

                    if isinstance(self.mask_layer.data, da.core.Array):
                        to_keep = np.unique(current_stack[self.mask_layer.data[i].compute() > 0])
                    else:
                        to_keep = np.unique(current_stack[self.mask_layer.data[i] > 0])

                    filtered_mask = functools.reduce(np.logical_or, (current_stack == val for val in to_keep))
                    filtered_data_tp = np.where(filtered_mask, current_stack, 0)

                    tifffile.imwrite(
                        os.path.join(
                            outputdir,
                            (
                                self.image1_layer.name
                                + "_filtered_labels_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        ),
                        np.array(filtered_data_tp, dtype="uint16"),
                    )

                file_list = [
                    os.path.join(outputdir, fname)
                    for fname in os.listdir(outputdir)
                    if fname.endswith(".tif")
                ]
                self.image1_layer = self.viewer.add_labels(
                    da.stack([imread(fname) for fname in sorted(file_list)]),
                    name=self.image1_layer.name + "_filtered_labels",
                )
            else:
                to_keep = np.unique(self.image1_layer.data[self.mask_layer.data > 0])
                filtered_mask = functools.reduce(np.logical_or, (self.image1_layer.data==val for val in to_keep))
                self.viewer.add_labels(np.where(filtered_mask, self.image1_layer.data, 0), name="selected labels")

        else:
            msg = QMessageBox()
            msg.setWindowTitle("Images do not have compatible shapes")
            msg.setText("Please provide images that have matching dimensions")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

    def delete_labels(self):
        """Delete labels that overlap with given mask. If the shape of the mask has 1 dimension less than the image, the mask will be applied to the current time point (index in the first dimension) of the image data."""

        if isinstance(self.mask_layer.data, da.core.Array):
            msg = QMessageBox()
            msg.setWindowTitle("Please provide a mask that is not a Dask array")
            msg.setText("Please provide a mask that is not a Dask array")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        # check data dimensions first
        image_shape = self.image1_layer.data.shape
        mask_shape = self.mask_layer.data.shape

        if len(image_shape) == len(mask_shape) + 1 and image_shape[1:] == mask_shape:
            # apply mask to single time point or to full stack depending on checkbox state
            if self.stack_checkbox.isChecked():
                # loop over all time points
                print('applying the mask to all time points')
                # check if the data is a dask array
                if isinstance(self.image1_layer.data, da.core.Array):
                    if self.outputdir is None:
                        self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

                    outputdir = os.path.join(
                        self.outputdir,
                        (self.image1_layer.name + "_filtered_labels"),
                    )
                    if os.path.exists(outputdir):
                        shutil.rmtree(outputdir)
                    os.mkdir(outputdir)

                    for i in range(
                        self.image1_layer.data.shape[0]
                    ):  # Loop over the first dimension
                        current_stack = self.image1_layer.data[
                            i
                        ].compute()  # Compute the current stack

                        to_delete = np.unique(current_stack[self.mask_layer.data > 0])
                        for label in to_delete:
                            current_stack[current_stack == label] = 0
                        tifffile.imwrite(
                            os.path.join(
                                outputdir,
                                (
                                    self.image1_layer.name
                                    + "_filtered_labels_TP"
                                    + str(i).zfill(4)
                                    + ".tif"
                                ),
                            ),
                            np.array(current_stack, dtype="uint16"),
                        )

                    file_list = [
                        os.path.join(outputdir, fname)
                        for fname in os.listdir(outputdir)
                        if fname.endswith(".tif")
                    ]
                    self.image1_layer = self.viewer.add_labels(
                        da.stack([imread(fname) for fname in sorted(file_list)]),
                        name=self.image1_layer.name + "_filtered_labels",
                    )

                else:
                    for tp in range(self.image1_layer.data.shape[0]):
                        to_delete = np.unique(self.image1_layer.data[tp][self.mask_layer.data > 0])
                        for label in to_delete:
                            self.image1_layer.data[tp][self.image1_layer.data[tp] == label] = 0

            else:
                tp = self.viewer.dims.current_step[0]
                print('applying the mask to the current time point only', tp)
                if isinstance(self.image1_layer.data, da.core.Array):
                    outputdir = QFileDialog.getExistingDirectory(self, "Please select the directory that holds the images. Data will be changed here. Selecting a new empty directory will create a copy of all data")

                    if len(os.listdir(outputdir)) == 0:
                        for i in range(
                            self.image1_layer.data.shape[0]
                        ):  # Loop over the first dimension
                            current_stack = self.image1_layer.data[
                                i
                            ].compute()  # Compute the current stack

                            if i == tp:
                                to_delete = np.unique(current_stack[self.mask_layer.data > 0])
                                for label in to_delete:
                                    current_stack[current_stack == label] = 0
                            tifffile.imwrite(
                                os.path.join(
                                    outputdir,
                                    (
                                        self.image1_layer.name
                                        + "_filtered_labels_TP"
                                        + str(i).zfill(4)
                                        + ".tif"
                                    ),
                                ),
                                np.array(current_stack, dtype="uint16"),
                            )

                            file_list = sorted([
                                os.path.join(outputdir, fname)
                                for fname in os.listdir(outputdir)
                                if fname.endswith(".tif")
                            ])
                    else:
                        current_stack = self.image1_layer.data[
                            tp
                        ].compute()  # Compute the current stack
                        to_delete = np.unique(current_stack[self.mask_layer.data > 0])
                        for label in to_delete:
                            current_stack[current_stack == label] = 0

                        file_list = sorted([
                            os.path.join(outputdir, fname)
                            for fname in os.listdir(outputdir)
                            if fname.endswith(".tif")
                        ])

                        tifffile.imwrite(
                            file_list[tp],
                            np.array(current_stack, dtype="uint16"),
                        )

                    self.image1_layer = self.viewer.add_labels(
                        da.stack([imread(fname) for fname in file_list]),
                        name=self.image1_layer.name + "_filtered_labels",
                    )

                else:
                    tp = self.viewer.dims.current_step[0]
                    to_delete = np.unique(self.image1_layer.data[tp][self.mask_layer.data > 0])
                    for label in to_delete:
                        self.image1_layer.data[tp][self.image1_layer.data[tp] == label] = 0

        elif image_shape == mask_shape:
            if isinstance(self.image1_layer.data, da.core.Array):
                if self.outputdir is None:
                    self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

                outputdir = os.path.join(
                    self.outputdir,
                    (self.image1_layer.name + "_filtered_labels"),
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(
                    self.image1_layer.data.shape[0]
                ):  # Loop over the first dimension
                    current_stack = self.image1_layer.data[
                        i
                    ].compute()  # Compute the current stack

                    to_delete = np.unique(current_stack[self.mask_layer.data[tp] > 0])
                    for label in to_delete:
                        current_stack[current_stack == label] = 0
                    tifffile.imwrite(
                        os.path.join(
                            outputdir,
                            (
                                self.image1_layer.name
                                + "_filtered_labels_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        ),
                        np.array(current_stack, dtype="uint16"),
                    )

                file_list = [
                    os.path.join(outputdir, fname)
                    for fname in os.listdir(outputdir)
                    if fname.endswith(".tif")
                ]
                self.image1_layer = self.viewer.add_labels(
                    da.stack([imread(fname) for fname in sorted(file_list)]),
                    name=self.image1_layer.name + "_filtered_labels",
                )
            else:
                to_delete = np.unique(self.image1_layer.data[self.mask_layer.data > 0])
                selected_labels = self.viewer.add_labels(copy.deepcopy(self.image1_layer.data), name="selected_self.image1_layer.data")
                for label in to_delete:
                    selected_labels.data[selected_labels.data == label] = 0

        else:
            msg = QMessageBox()
            msg.setWindowTitle("Images do not have compatible shapes")
            msg.setText("Please provide images that have matching dimensions")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
