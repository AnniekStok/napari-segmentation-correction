import functools
import os
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
)
from skimage.io import imread

from napari_segmentation_correction.helpers.base_tool_widget import BaseToolWidget
from napari_segmentation_correction.helpers.process_actions_helpers import (
    remove_invalid_chars,
)


class PropertyFilterWidget(BaseToolWidget):
    """Widget to filter objects by numerical property"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(napari.layers.Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        box = QGroupBox("Filter objects by property")
        filter_layout = QHBoxLayout()

        self.property = QComboBox()
        self.property.currentIndexChanged.connect(self.update_min_max_value)
        self.update_status.connect(self.set_properties)
        self.operation = QComboBox()
        self.operation.addItems([">", "<", ">=", "<="])
        self.operation.setToolTip("Operation to apply for filtering")
        self.value = QDoubleSpinBox()
        self.value.setValue(0.0)
        self.value.setSingleStep(0.1)
        self.value.setMaximum(10e6)
        self.value.setMinimum(-10e6)
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

        box.setLayout(main_layout)

        layout = QVBoxLayout()
        layout.addWidget(box)
        self.setLayout(layout)

    def set_properties(self) -> None:
        """Set available properties for the selected layer"""

        current_prop = self.property.currentText()
        if self.layer is not None:
            props = list(self.layer.properties.keys())
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
        if prop in self.layer.properties:
            values = self.layer.properties[prop]
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

        if self.layer is not None:
            if isinstance(self.layer.data, da.core.Array):
                outputdir = QFileDialog.getExistingDirectory(
                    self, "Select Output Folder"
                )
                if not outputdir:
                    return

                outputdir = os.path.join(
                    outputdir,
                    remove_invalid_chars(self.layer.name + "_filtered"),
                )
                while os.path.exists(outputdir):
                    outputdir = outputdir + "_1"
                os.mkdir(outputdir)

            df = pd.DataFrame(self.layer.properties)
            if "time_point" in df.columns and prop in df.columns:
                filtered_label_imgs = []
                for time_point in range(self.layer.data.shape[0]):
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

                    if isinstance(self.layer.data, da.core.Array):
                        labels = np.array(self.layer.data[time_point].compute())
                    else:
                        labels = np.array(self.layer.data[time_point])

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

                    if isinstance(self.layer.data, da.core.Array):
                        tifffile.imwrite(
                            os.path.join(
                                outputdir,
                                (
                                    self.layer.name
                                    + "_filtered_TP"
                                    + str(time_point).zfill(4)
                                    + ".tif"
                                ),
                            ),
                            np.array(new_labels, dtype="uint16"),
                        )
                    else:
                        filtered_label_imgs.append(new_labels)

                if isinstance(self.layer.data, da.core.Array):
                    file_list = [
                        os.path.join(outputdir, fname)
                        for fname in os.listdir(outputdir)
                        if fname.endswith(".tif")
                    ]
                    self.layer = self.viewer.add_labels(
                        da.stack([imread(fname) for fname in sorted(file_list)]),
                        name=self.layer.name + "_filtered",
                        scale=self.layer.scale,
                    )
                else:
                    result = np.stack(filtered_label_imgs)
                    self.layer = self.viewer.add_labels(
                        result,
                        name=self.layer.name + "_filtered",
                        scale=self.layer.scale,
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

                labels = np.array(self.layer.data)
                mask = np.isin(labels, filtered_labels)
                if keep_delete == "Delete":
                    result = np.where(~mask, labels, 0)
                else:
                    result = np.where(mask, labels, 0)

                self.layer = self.viewer.add_labels(
                    result,
                    name=self.layer.name + "_filtered",
                    scale=self.layer.scale,
                )

            else:
                warn(f"Property {prop} not found in layer properties", stacklevel=2)
                return
