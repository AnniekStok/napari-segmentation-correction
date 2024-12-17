import functools
import os
import shutil
from warnings import warn

import dask.array as da
import napari
import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage import measure
from skimage.io import imread

from .layer_manager import LayerManager


class SizeFilterWidget(QWidget):
    """Widget to filter objects by size (pixels)"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        filterbox = QGroupBox("Filter objects by size")
        filter_layout = QVBoxLayout()

        label_size = QLabel("Size threshold (voxels)")
        threshold_size_layout = QHBoxLayout()
        self.min_size_field = QSpinBox()
        self.min_size_field.setMaximum(10000000)
        self.delete_btn = QPushButton("Delete")
        threshold_size_layout.addWidget(self.min_size_field)
        threshold_size_layout.addWidget(self.delete_btn)

        filter_layout.addWidget(label_size)
        filter_layout.addLayout(threshold_size_layout)
        self.delete_btn.clicked.connect(self._delete_small_objects)
        self.delete_btn.setEnabled(True)

        filterbox.setLayout(filter_layout)

        layout = QVBoxLayout()
        layout.addWidget(filterbox)
        self.setLayout(layout)

    def _delete_small_objects(self) -> None:
        """Delete small objects in the selected layer"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_sizefiltered"),
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

                # measure the sizes in pixels of the labels in slice using skimage.regionprops
                props = measure.regionprops(current_stack)
                filtered_labels = [
                    p.label
                    for p in props
                    if p.num_pixels > self.min_size_field.value()
                ]
                mask = functools.reduce(
                    np.logical_or,
                    (current_stack == val for val in filtered_labels),
                )
                filtered = np.where(mask, current_stack, 0)
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_sizefiltered_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(filtered, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name
                + "_sizefiltered",
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )

        else:
            # Image data is a normal array and can be directly edited.
            if len(self.label_manager.selected_layer.data.shape) == 4:
                stack = []
                for i in range(
                    self.label_manager.selected_layer.data.shape[0]
                ):
                    props = measure.regionprops(
                        self.label_manager.selected_layer.data[i]
                    )
                    filtered_labels = [
                        p.label
                        for p in props
                        if p.num_pixels > self.min_size_field.value()
                    ]
                    mask = functools.reduce(
                        np.logical_or,
                        (
                            self.label_manager.selected_layer.data[i] == val
                            for val in filtered_labels
                        ),
                    )
                    filtered = np.where(
                        mask, self.label_manager.selected_layer.data[i], 0
                    )
                    stack.append(filtered)
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.stack(stack, axis=0),
                    name=self.label_manager.selected_layer.name
                    + "_sizefiltered",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )

            elif len(self.label_manager.selected_layer.data.shape) in (2, 3):
                props = measure.regionprops(
                    self.label_manager.selected_layer.data
                )
                filtered_labels = [
                    p.label
                    for p in props
                    if p.num_pixels > self.min_size_field.value()
                ]

                if len(filtered_labels) == 0:
                    warn(f"No labels are larger than {self.min_size_field.value()}", stacklevel=2)
                    return None

                mask = functools.reduce(
                    np.logical_or,
                    (
                        self.label_manager.selected_layer.data == val
                        for val in filtered_labels
                    ),
                )
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.where(mask, self.label_manager.selected_layer.data, 0),
                    name=self.label_manager.selected_layer.name
                    + "_sizefiltered",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )

            else:
                print("length of input shape should be 2, 3, or 4")
