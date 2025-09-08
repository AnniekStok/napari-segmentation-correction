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
    QWidget,
)

from .layer_manager import LayerManager


class SaveLabelsWidget(QWidget):
    """Widget for saving label data with options for datatype, compression, and whether to split time points."""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager

        # Select datatype
        self.select_dtype = QComboBox()
        self.select_dtype.addItem("np.uint8")
        self.select_dtype.addItem("np.uint16")
        self.select_dtype.addItem("np.uint32")
        self.select_dtype.addItem("np.uint64")
        self.select_dtype.setToolTip("File bit depth for saving.")

        # Split time points
        self.split_time_points = QCheckBox("Split time points")
        self.split_time_points.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
            and self.label_manager.selected_layer.data.ndim >= 3
        )
        self.label_manager.layer_update.connect(
            lambda: self.split_time_points.setEnabled(
                isinstance(self.label_manager.selected_layer, napari.layers.Labels)
                and self.label_manager.selected_layer.data.ndim >= 3
            )
        )
        self.split_time_points.setToolTip(
            "Saves each time point to a separate file. Assumes that the time dimension is along the first axis."
        )

        # Use compression
        self.use_compression = QCheckBox("Use compression")
        self.use_compression.setChecked(True)
        self.use_compression.setToolTip(
            "Use zstd compression? This may take a bit longer to save."
        )

        # Filename
        self.filename = QLineEdit()
        self.filename.setPlaceholderText("File name")
        self.label_manager.layer_update.connect(self.update_filename)
        self.filename.setToolTip("Filename for saving labels")

        ## Add save button
        self.save_btn = QPushButton("Save labels")
        self.save_btn.clicked.connect(self._save_labels)
        self.save_btn.setEnabled(isinstance(self.label_manager.selected_layer, Labels))
        self.label_manager.layer_update.connect(
            lambda: self.save_btn.setEnabled(
                isinstance(self.label_manager.selected_layer, napari.layers.Labels)
            )
        )

        # Combine layouts
        save_box = QGroupBox("Save labels")
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

    def update_filename(self) -> None:
        """Update the default filename"""

        if isinstance(self.label_manager.selected_layer, napari.layers.Labels):
            self.filename.setText(self.label_manager.selected_layer.name)

    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        data = self.label_manager.selected_layer.data
        ndim = data.ndim
        split_time_points = ndim >= 3 and self.split_time_points.isChecked()
        dtype_map = {
            "np.uint8": np.uint8,
            "np.uint16": np.uint16,
            "np.uint32": np.uint32,
            "np.uint64": np.uint64,
        }
        dtype = dtype_map[self.select_dtype.currentText()]
        use_compression = self.use_compression.isChecked()
        filename = self.filename.text()

        destination = QFileDialog.getExistingDirectory(self, "Select Output Folder")

        if ndim >= 3 and split_time_points:
            for i in range(data.shape[0]):
                if isinstance(data, da.core.Array):
                    current_stack = data[i].compute().astype(dtype)
                else:
                    current_stack = data[i].astype(dtype)

                tifffile.imwrite(
                    (
                        os.path.join(
                            destination,
                            (
                                filename.split(".tif")[0]
                                + "_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        )
                    ),
                    current_stack,
                    compression="zstd" if use_compression else None,
                )

        else:
            tifffile.imwrite(
                os.path.join(destination, (filename + ".tif")),
                data.astype(dtype),
                compression="zstd" if use_compression else None,
            )
