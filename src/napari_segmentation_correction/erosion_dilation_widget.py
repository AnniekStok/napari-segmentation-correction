import os
import shutil

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
from scipy import ndimage
from scipy.ndimage import binary_erosion
from skimage.io import imread
from skimage.segmentation import expand_labels

from .layer_manager import LayerManager


class ErosionDilationWidget(QWidget):
    """Widget to perform erosion/dilation on label images"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        dil_erode_box = QGroupBox("Erode/dilate labels")
        dil_erode_box_layout = QVBoxLayout()

        radius_layout = QHBoxLayout()
        str_element_diameter_label = QLabel("Structuring element diameter")
        str_element_diameter_label.setFixedWidth(200)
        self.structuring_element_diameter = QSpinBox()
        self.structuring_element_diameter.setMaximum(100)
        self.structuring_element_diameter.setValue(1)
        radius_layout.addWidget(str_element_diameter_label)
        radius_layout.addWidget(self.structuring_element_diameter)

        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Iterations")
        iterations_label.setFixedWidth(200)
        self.iterations = QSpinBox()
        self.iterations.setMaximum(100)
        self.iterations.setValue(1)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations)

        shrink_dilate_buttons_layout = QHBoxLayout()
        self.erode_btn = QPushButton("Erode")
        self.dilate_btn = QPushButton("Dilate")
        self.erode_btn.clicked.connect(self._erode_labels)
        self.dilate_btn.clicked.connect(self._dilate_labels)
        shrink_dilate_buttons_layout.addWidget(self.erode_btn)
        shrink_dilate_buttons_layout.addWidget(self.dilate_btn)

        if self.label_manager.selected_layer is not None:
            self.erode_btn.setEnabled(True)
            self.dilate_btn.setEnabled(True)

        dil_erode_box_layout.addLayout(radius_layout)
        dil_erode_box_layout.addLayout(iterations_layout)
        dil_erode_box_layout.addLayout(shrink_dilate_buttons_layout)

        dil_erode_box.setLayout(dil_erode_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(dil_erode_box)
        self.setLayout(layout)

    def _erode_labels(self):
        """Shrink oversized labels through erosion"""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()
        structuring_element = np.ones(
            (diam, diam, diam), dtype=bool
        )  # Define a 3x3x3 structuring element for 3D erosion

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_eroded"),
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
                mask = current_stack > 0
                filled_mask = ndimage.binary_fill_holes(mask)
                eroded_mask = binary_erosion(
                    filled_mask,
                    structure=structuring_element,
                    iterations=iterations,
                )
                eroded = np.where(eroded_mask, current_stack, 0)
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_eroded_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(eroded, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name + "_eroded",
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )
            return True

        else:
            if len(self.label_manager.selected_layer.data.shape) == 4:
                stack = []
                for i in range(
                    self.label_manager.selected_layer.data.shape[0]
                ):
                    mask = self.label_manager.selected_layer.data[i] > 0
                    filled_mask = ndimage.binary_fill_holes(mask)
                    eroded_mask = binary_erosion(
                        filled_mask,
                        structure=structuring_element,
                        iterations=iterations,
                    )
                    stack.append(
                        np.where(
                            eroded_mask,
                            self.label_manager.selected_layer.data[i],
                            0,
                        )
                    )
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.stack(stack, axis=0),
                    name=self.label_manager.selected_layer.name + "_eroded",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )
            elif len(self.label_manager.selected_layer.data.shape) == 3:
                mask = self.label_manager.selected_layer.data > 0
                filled_mask = ndimage.binary_fill_holes(mask)
                eroded_mask = binary_erosion(
                    filled_mask,
                    structure=structuring_element,
                    iterations=iterations,
                )
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.where(
                        eroded_mask, self.label_manager.selected_layer.data, 0
                    ),
                    name=self.label_manager.selected_layer.name + "_eroded",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )
            else:
                print("4D or 3D array required!")

    def _dilate_labels(self):
        """Dilate labels in the selected layer."""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_dilated"),
            )
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
            os.mkdir(outputdir)

            for i in range(
                self.label_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                expanded_labels = self.label_manager.selected_layer.data[
                    i
                ].compute()  # Compute the current stack
                for _j in range(iterations):
                    expanded_labels = expand_labels(
                        expanded_labels, distance=diam
                    )
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_dilated_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(expanded_labels, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name + "_dilated",
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )
            return True

        else:
            if len(self.label_manager.selected_layer.data.shape) == 4:
                stack = []
                for i in range(
                    self.label_manager.selected_layer.data.shape[0]
                ):
                    expanded_labels = self.label_manager.selected_layer.data[i]
                    for _j in range(iterations):
                        expanded_labels = expand_labels(
                            expanded_labels, distance=diam
                        )
                    stack.append(expanded_labels)
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.stack(stack, axis=0),
                    name=self.label_manager.selected_layer.name + "_dilated",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )

            elif len(self.label_manager.selected_layer.data.shape) == 3:
                expanded_labels = self.label_manager.selected_layer.data
                for _i in range(iterations):
                    expanded_labels = expand_labels(
                        expanded_labels, distance=diam
                    )

                self.label_manager.selected_layer = self.viewer.add_labels(
                    expanded_labels,
                    name=self.label_manager.selected_layer.name + "_dilated",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )
            else:
                print("input should be a 3D or 4D stack")
