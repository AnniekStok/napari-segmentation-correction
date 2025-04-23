import os
import shutil

import dask.array as da
import napari
import numpy as np
import tifffile
from napari.layers import Image, Labels
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
from skimage.io import imread

from .layer_dropdown import LayerDropdown


class ThresholdWidget(QWidget):
    """Widget that applies a threshold to an image or labels layer"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer
        self.outputdir = None
        self.threshold_layer = None

        threshold_box = QGroupBox("Threshold")
        threshold_box_layout = QVBoxLayout()

        self.threshold_layer_dropdown = LayerDropdown(
            self.viewer, (Image, Labels)
        )
        self.threshold_layer_dropdown.layer_changed.connect(
            self._update_threshold_layer
        )
        threshold_box_layout.addWidget(self.threshold_layer_dropdown)

        min_threshold_layout = QHBoxLayout()
        min_threshold_layout.addWidget(QLabel("Min value"))
        self.min_threshold = QSpinBox()
        self.min_threshold.setMaximum(65535)
        min_threshold_layout.addWidget(self.min_threshold)

        max_threshold_layout = QHBoxLayout()
        max_threshold_layout.addWidget(QLabel("Max value"))
        self.max_threshold = QSpinBox()
        self.max_threshold.setMaximum(65535)
        self.max_threshold.setValue(65535)
        max_threshold_layout.addWidget(self.max_threshold)

        threshold_box_layout.addLayout(min_threshold_layout)
        threshold_box_layout.addLayout(max_threshold_layout)
        threshold_btn = QPushButton("Run")
        threshold_btn.clicked.connect(self._threshold)
        threshold_box_layout.addWidget(threshold_btn)

        threshold_box.setLayout(threshold_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(threshold_box)
        self.setLayout(layout)

    def _update_threshold_layer(self, selected_layer) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.threshold_layer = None
        else:
            self.threshold_layer = self.viewer.layers[selected_layer]
            self.threshold_layer_dropdown.setCurrentText(selected_layer)

    def _threshold(self):
        """Threshold the selected label or intensity image"""

        if isinstance(self.threshold_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

            outputdir = os.path.join(
                self.outputdir,
                (self.threshold_layer.name + "_threshold"),
            )
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
            os.mkdir(outputdir)

            for i in range(
                self.threshold_layer.data.shape[0]
            ):  # Loop over the first dimension
                data = self.threshold_layer.data[
                    i
                ].compute()  # Compute the current stack

                thresholded = (
                    data >= int(self.min_threshold.value())
                ) & (data <= int(self.max_threshold.value()))

                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.threshold_layer.name
                            + "_thresholded_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(thresholded, dtype="uint8"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.threshold_layer.name + "_thresholded",
                scale=self.threshold_layer.scale
            )

        else:
            thresholded = (
                self.threshold_layer.data >= int(self.min_threshold.value())
            ) & (self.threshold_layer.data <= int(self.max_threshold.value()))
            self.viewer.add_labels(
                thresholded, name=self.threshold_layer.name + "_thresholded",
                scale=self.threshold_layer.scale
            )
