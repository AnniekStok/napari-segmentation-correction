import functools
import os
import shutil
from warnings import warn

import dask.array as da
import napari
import numpy as np
import pandas as pd
import tifffile
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread

from .layer_manager import LayerManager


class PropertyFilterWidget(QWidget):
    """Widget to filter objects by numerical property"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        filterbox = QGroupBox("Filter objects by property")
        filter_layout = QHBoxLayout()

        self.property = QComboBox()
        self.property.currentIndexChanged.connect(self.update_min_max_value)
        self.label_manager.layer_update.connect(self.set_properties)
        self.operation = QComboBox()
        self.operation.addItems([">", "<", ">=", "<="])
        self.operation.setToolTip("Operation to apply for filtering")
        self.value = QDoubleSpinBox()
        self.value.setValue(0.0)
        self.value.setSingleStep(0.1)
        self.value.setToolTip("Threshold value for the selected property")
        self.keep_delete = QComboBox()
        self.keep_delete.addItems(["Keep", "Delete"])
        self.keep_delete.setToolTip(
            "Choose whether to keep or delete objects matching the criteria"
        )
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.filter_by_property)

        filter_layout.addWidget(self.property)
        filter_layout.addWidget(self.value)
        filter_layout.addWidget(self.operation)
        filter_layout.addWidget(self.keep_delete)

        main_layout = QVBoxLayout()
        main_layout.addLayout(filter_layout)
        main_layout.addWidget(self.run_btn)

        filterbox.setLayout(main_layout)

        layout = QVBoxLayout()
        layout.addWidget(filterbox)
        self.setLayout(layout)

    def set_properties(self) -> None:
        """Set available properties for the selected layer"""
        current_prop = self.property.currentText()
        if self.label_manager.selected_layer is not None:
            props = list(self.label_manager.selected_layer.properties.keys())
            self.property.clear()
            self.property.addItems(
                [p for p in props if p not in ("label", "time_point")]
            )
            if current_prop in props:
                self.property.setCurrentText(current_prop)
            self.run_btn.setEnabled(True) if len(
                props
            ) > 0 else self.run_btn.setEnabled(False)
        else:
            self.run_btn.setEnabled(False)

    def update_min_max_value(self) -> None:
        """Update min and max values for the threshold spinbox based on selected property"""
        prop = self.property.currentText()
        if prop in self.label_manager.selected_layer.properties:
            values = self.label_manager.selected_layer.properties[prop]
            self.value.setMinimum(np.min(values))
            self.value.setMaximum(np.max(values))
            if self.value.value() < np.min(values) or self.value.value() > np.max(
                values
            ):
                self.value.setValue(np.min(values))

    def filter_by_property(self) -> None:
        """Filter objects by selected property and threshold value"""

        prop = self.property.currentText()
        value = self.value.value()
        operation = self.operation.currentText()
        keep_delete = self.keep_delete.currentText()

        if self.label_manager.selected_layer is not None:
            if isinstance(self.label_manager.selected_layer.data, da.core.Array):
                if self.outputdir is None:
                    self.outputdir = QFileDialog.getExistingDirectory(
                        self, "Select Output Folder"
                    )

                outputdir = os.path.join(
                    self.outputdir,
                    (self.label_manager.selected_layer.name + "_filtered"),
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

            df = pd.DataFrame(self.label_manager.selected_layer.properties)
            if "time_point" in df.columns and prop in df.columns:
                filtered_label_imgs = []
                for time_point in range(
                    self.label_manager.selected_layer.data.shape[0]
                ):
                    df_subset = df.loc[df["time_point"] == time_point]

                    if operation == ">":
                        filtered_labels = df_subset.loc[
                            df_subset[prop] > value, "label"
                        ]
                    elif operation == "<":
                        filtered_labels = df_subset.loc[
                            df_subset[prop] < value, "label"
                        ]
                    elif operation == "<=":
                        filtered_labels = df_subset.loc[
                            df_subset[prop] <= value, "label"
                        ]
                    elif operation == ">=":
                        filtered_labels = df_subset.loc[
                            df_subset[prop] >= value, "label"
                        ]

                    if isinstance(self.label_manager.selected_layer.data, da.Array):
                        labels = np.array(
                            self.label_manager.selected_layer.data[time_point].compute()
                        )
                    else:
                        labels = np.array(
                            self.label_manager.selected_layer.data[time_point]
                        )

                    if len(filtered_labels) == 0:
                        mask = np.zeros_like(labels, dtype=bool)
                    else:
                        mask = functools.reduce(
                            np.logical_or, (labels == val for val in filtered_labels)
                        )
                    if keep_delete == "Delete":
                        new_labels = np.where(~mask, labels, 0)
                    else:
                        new_labels = np.where(mask, labels, 0)

                    if isinstance(self.label_manager.selected_layer.data, da.Array):
                        tifffile.imwrite(
                            os.path.join(
                                outputdir,
                                (
                                    self.label_manager.selected_layer.name
                                    + "_filtered_TP"
                                    + str(time_point).zfill(4)
                                    + ".tif"
                                ),
                            ),
                            np.array(new_labels, dtype="uint16"),
                        )
                    else:
                        filtered_label_imgs.append(new_labels)

                if isinstance(self.label_manager.selected_layer.data, da.Array):
                    file_list = [
                        os.path.join(outputdir, fname)
                        for fname in os.listdir(outputdir)
                        if fname.endswith(".tif")
                    ]
                    self.label_manager.selected_layer = self.viewer.add_labels(
                        da.stack([imread(fname) for fname in sorted(file_list)]),
                        name=self.label_manager.selected_layer.name + "_filtered",
                        scale=self.label_manager.selected_layer.scale,
                    )
                else:
                    result = np.stack(filtered_label_imgs)
                    self.label_manager.selected_layer = self.viewer.add_labels(
                        result,
                        name=self.label_manager.selected_layer.name + "_filtered",
                        scale=self.label_manager.selected_layer.scale,
                    )

            elif prop in df.columns:
                if operation == ">":
                    filtered_labels = df.loc[df[prop] > value, "label"]
                elif operation == "<":
                    filtered_labels = df.loc[df[prop] < value, "label"]
                elif operation == "<=":
                    filtered_labels = df.loc[df[prop] <= value, "label"]
                elif operation == ">=":
                    filtered_labels = df.loc[df[prop] >= value, "label"]

                labels = np.array(self.label_manager.selected_layer.data)
                mask = functools.reduce(
                    np.logical_or,
                    (labels == val for val in filtered_labels),
                )
                if keep_delete == "Delete":
                    result = np.where(~mask, labels, 0)
                else:
                    result = np.where(mask, labels, 0)

                self.label_manager.selected_layer = self.viewer.add_labels(
                    result,
                    name=self.label_manager.selected_layer.name + "_filtered",
                    scale=self.label_manager.selected_layer.scale,
                )

            else:
                warn(f"Property {prop} not found in layer properties", stacklevel=2)
                return
