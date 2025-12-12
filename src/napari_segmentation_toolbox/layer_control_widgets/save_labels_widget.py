import os

import dask.array as da
import napari
import numpy as np
import tifffile
from napari.layers import Labels
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from napari_segmentation_toolbox.helpers.base_tool_widget import BaseToolWidget
from napari_segmentation_toolbox.helpers.process_actions_helpers import (
    remove_invalid_chars,
)


class SaveLabelsWidget(BaseToolWidget):
    """Widget for saving label data with options for datatype, compression, and whether to split time points."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        layer_type=(napari.layers.Labels, napari.layers.Image),
    ) -> None:
        super().__init__(viewer, layer_type)

        # Select datatype
        self.select_dtype = QComboBox()
        self.select_dtype.addItem("np.int8")
        self.select_dtype.addItem("np.int16")
        self.select_dtype.addItem("np.int32")
        self.select_dtype.addItem("np.int64")
        self.select_dtype.addItem("np.uint8")
        self.select_dtype.addItem("np.uint16")
        self.select_dtype.addItem("np.uint32")
        self.select_dtype.addItem("np.uint64")
        self.select_dtype.addItem("np.float32")
        self.select_dtype.addItem("np.float64")
        self.select_dtype.setToolTip("File bit depth for saving.")
        self.select_dtype.setCurrentIndex(5)

        # Split time points
        self.split_time_points = QCheckBox("Split time points")
        self.split_time_points.setEnabled(
            isinstance(self.layer, napari.layers.Labels | napari.layers.Image)
            and self.layer.data.ndim >= 3
        )
        self.split_time_points.setToolTip(
            "Saves each time point to a separate file. Assumes that the time dimension is along the first axis."
        )

        # Use compression
        self.use_compression = QCheckBox("Use compression")
        self.use_compression.setChecked(True)
        self.use_compression.setToolTip(
            "Use 'deflate' compression? This may take a bit longer to save."
        )

        # Filename
        self.filename = QLineEdit()
        self.filename.setPlaceholderText("File name")

        ## Add save button
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_labels)
        self.save_btn.setEnabled(isinstance(self.layer, Labels))

        # Combine layouts
        save_box = QGroupBox("Save")
        layout = QVBoxLayout()

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(self.select_dtype)
        settings_layout.addWidget(self.split_time_points)
        settings_layout.addWidget(self.use_compression)
        layout.addLayout(settings_layout)
        layout.addWidget(self.filename)
        layout.addWidget(self.save_btn)
        save_box.setLayout(layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(save_box)
        self.setLayout(main_layout)

        self.update_status.connect(self._update_status)

    def _update_status(self) -> None:
        """Update the button status and filename"""

        active = self.layer is not None

        self.split_time_points.setEnabled(active and self.layer.data.ndim >= 3)
        self.save_btn.setEnabled(active)
        if active:
            self.filename.setText(self.layer.name)

    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        data = self.layer.data
        ndim = data.ndim
        split_time_points = ndim >= 3 and self.split_time_points.isChecked()
        dtype_map = {
            "np.int8": np.int8,
            "np.int16": np.int16,
            "np.int32": np.int32,
            "np.int64": np.int64,
            "np.uint8": np.uint8,
            "np.uint16": np.uint16,
            "np.uint32": np.uint32,
            "np.uint64": np.uint64,
            "np.float32": np.float32,
            "np.float64": np.float64,
        }
        dtype = dtype_map[self.select_dtype.currentText()]
        use_compression = self.use_compression.isChecked()
        filename = remove_invalid_chars(self.filename.text())

        outputdir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if not outputdir:
            return

        if ndim >= 3 and split_time_points:
            for i in range(data.shape[0]):
                if isinstance(data, da.core.Array):
                    current_stack = data[i].compute().astype(dtype)
                else:
                    current_stack = data[i].astype(dtype)

                tifffile.imwrite(
                    (
                        os.path.join(
                            outputdir,
                            (
                                filename.split(".tif")[0]
                                + "_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        )
                    ),
                    current_stack,
                    compression="deflate" if use_compression else None,
                )

        else:
            tifffile.imwrite(
                os.path.join(outputdir, (filename + ".tif")),
                data.astype(dtype),
                compression="deflate" if use_compression else None,
            )
