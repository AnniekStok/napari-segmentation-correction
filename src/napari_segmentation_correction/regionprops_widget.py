import dask.array as da
import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from .custom_table_widget import ColoredTableWidget
from .layer_dropdown import LayerDropdown
from .layer_manager import LayerManager
from .prop_filter_widget import PropertyFilterWidget
from .regionprops_extended import calculate_extended_props


class RegionPropsWidget(QWidget):
    """Widget showing region props as a table and plot widget"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.table = None
        self.ndims = 2
        self.feature_dims = 2
        self.axis_widgets = []

        # features widget
        self.feature_properties = [
            {
                "region_prop_name": "intensity_mean",
                "display_name": "Mean intensity",
                "enabled": False,
                "selected": False,
                "dims": [2, 3],
            },
            {
                "region_prop_name": "area",
                "display_name": "Area",
                "enabled": True,
                "selected": True,
                "dims": [2],
            },
            {
                "region_prop_name": "perimeter",
                "display_name": "Perimeter",
                "enabled": True,
                "selected": False,
                "dims": [2],
            },
            {
                "region_prop_name": "circularity",
                "display_name": "Circularity",
                "enabled": True,
                "selected": False,
                "dims": [2],
            },
            {
                "region_prop_name": "ellipse_axes",
                "display_name": "Ellipse axes",
                "enabled": True,
                "selected": False,
                "dims": [2],
            },
            {
                "region_prop_name": "volume",
                "display_name": "Volume",
                "enabled": True,
                "selected": True,
                "dims": [3],
            },
            {
                "region_prop_name": "surface_area",
                "display_name": "Surface area",
                "enabled": True,
                "selected": False,
                "dims": [3],
            },
            {
                "region_prop_name": "sphericity",
                "display_name": "Sphericity",
                "enabled": True,
                "selected": False,
                "dims": [3],
            },
            {
                "region_prop_name": "ellipsoid_axes",
                "display_name": "Ellipsoid axes",
                "enabled": True,
                "selected": False,
                "dims": [3],
            },
        ]

        feature_box = QGroupBox("Features to measure")
        feature_box.setMaximumHeight(250)
        self.checkbox_layout = QVBoxLayout()
        self.checkboxes = []
        self.intensity_image_dropdown = None

        feature_box.setLayout(self.checkbox_layout)

        # Push button to measure features
        self.measure_btn = QPushButton("Measure properties")
        self.measure_btn.clicked.connect(self.measure)
        self.measure_btn.setEnabled(False)

        ### Add widget for property filtering
        self.prop_filter_widget = PropertyFilterWidget(self.viewer, self.label_manager)
        self.prop_filter_widget.setVisible(False)

        # Add table layout
        self.regionprops_layout = QVBoxLayout()

        # Assemble layout
        main_box = QGroupBox("Region properties")
        main_layout = QVBoxLayout()
        main_layout.addWidget(feature_box)
        main_layout.addWidget(self.measure_btn)
        main_layout.addWidget(self.prop_filter_widget)
        main_layout.addLayout(self.regionprops_layout)
        main_box.setLayout(main_layout)

        layout = QVBoxLayout()
        layout.addWidget(main_box)

        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)

        # connect to update signal
        self.label_manager.layer_update.connect(self.update_properties)

    def update_properties(self) -> None:
        """Update the available properties based on the selected label layer dimensions"""

        if self.label_manager.selected_layer is None:
            self.measure_btn.setEnabled(False)
            return
        if (
            self.label_manager.selected_layer is not None
            and "dimension_info" in self.label_manager.selected_layer.metadata
        ):
            _, axes_labels, _ = self.label_manager.selected_layer.metadata[
                "dimension_info"
            ]
            self.feature_dims = 3 if "Z" in axes_labels else 2
        else:
            self.feature_dims = 2

        self.measure_btn.setEnabled(True)

        if hasattr(self, "intensity_image_dropdown") and self.intensity_image_dropdown:
            self.intensity_image_dropdown.layer_changed.disconnect()
            self.intensity_image_dropdown.deleteLater()
        while self.checkbox_layout.count():
            item = self.checkbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.intensity_image_dropdown = None
        self.checkboxes = []

        # create checkbox for each feature
        self.properties = [
            f for f in self.feature_properties if self.feature_dims in f["dims"]
        ]
        self.checkbox_state = {
            prop["region_prop_name"]: prop["selected"] for prop in self.properties
        }
        for prop in self.properties:
            if self.feature_dims in prop["dims"]:
                checkbox = QCheckBox(prop["display_name"])
                checkbox.setEnabled(prop["enabled"])
                checkbox.setStyleSheet("QCheckBox:disabled { color: grey }")
                checkbox.setChecked(self.checkbox_state[prop["region_prop_name"]])
                checkbox.stateChanged.connect(
                    lambda state, prop=prop: self.checkbox_state.update(
                        {prop["region_prop_name"]: state == 2}
                    )
                )
                self.checkboxes.append(
                    {"region_prop_name": prop["region_prop_name"], "checkbox": checkbox}
                )

                if prop["region_prop_name"] == "intensity_mean":
                    self.intensity_image_dropdown = LayerDropdown(
                        self.viewer, napari.layers.Image
                    )
                    self.intensity_image_dropdown.layer_changed.connect(
                        self._update_intensity_checkbox
                    )
                    if self.intensity_image_dropdown.selected_layer is not None:
                        checkbox.setEnabled(True)
                    int_layout = QHBoxLayout()
                    int_layout.addWidget(checkbox)
                    int_layout.addWidget(self.intensity_image_dropdown)
                    int_layout.setContentsMargins(0, 0, 0, 0)
                    int_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                    int_widget = QWidget()
                    int_widget.setLayout(int_layout)

                    # Enforce same height behavior as a normal checkbox
                    int_widget.setSizePolicy(checkbox.sizePolicy())
                    self.intensity_image_dropdown.setSizePolicy(
                        QSizePolicy.Expanding, QSizePolicy.Fixed
                    )

                    self.checkbox_layout.addWidget(int_widget)
                else:
                    self.checkbox_layout.addWidget(checkbox)

    def _update_intensity_checkbox(self) -> None:
        """Enable or disable the intensity_mean checkbox based on the selected layer."""
        if self.intensity_image_dropdown is not None:
            checkbox = next(
                (
                    cb["checkbox"]
                    for cb in self.checkboxes
                    if cb["region_prop_name"] == "intensity_mean"
                ),
                None,
            )
            if checkbox is not None:
                checkbox.setEnabled(
                    isinstance(
                        self.intensity_image_dropdown.selected_layer,
                        napari.layers.Image,
                    )
                )

    def measure(self):
        """Measure the selected region properties and update the table."""

        _, axes_labels, spacing = self.label_manager.selected_layer.metadata[
            "dimension_info"
        ]
        self.use_z = "Z" in axes_labels
        self.ndims = len(axes_labels)
        spacing = [
            s
            for s, label in zip(spacing, axes_labels, strict=False)
            if label not in ("C", "T")
        ]
        features = self.get_selected_features()

        if (
            isinstance(
                self.intensity_image_dropdown.selected_layer, napari.layers.Image
            )
            and "intensity_mean" in features
        ):
            intensity_image = self.intensity_image_dropdown.selected_layer.data
        else:
            intensity_image = None

        data = self.label_manager.selected_layer.data
        if intensity_image is not None and intensity_image.shape != data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Shape mismatch")
            msg.setText(
                f"Label layer and intensity image must have the same shape. Got {self.label_manager.selected_layer.data.shape} and {intensity_image.shape}."
            )
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        if (self.use_z and self.ndims == 3) or (not self.use_z and self.ndims == 2):
            props = calculate_extended_props(
                data,
                intensity_image=intensity_image,
                properties=features,
                spacing=spacing,
            )
        else:
            for i in tqdm(range(data.shape[0])):
                d = data[i].compute() if isinstance(data, da.core.Array) else data[i]
                if isinstance(intensity_image, da.core.Array):
                    int_img = intensity_image[i].compute()
                elif isinstance(intensity_image, np.ndarray):
                    int_img = intensity_image[i]
                else:
                    int_img = None
                props_slice = calculate_extended_props(
                    d,
                    intensity_image=int_img,
                    properties=features,
                    spacing=spacing,
                )
                props_slice["time_point"] = i
                if i == 0:
                    props = props_slice
                else:
                    props = pd.concat([props, props_slice], ignore_index=True)

        if hasattr(self.label_manager.selected_layer, "properties"):
            self.label_manager.selected_layer.properties = props
            self.prop_filter_widget.set_properties()
            self.update_table()

    def update_table(self) -> None:
        """Update the regionprops table based on the selected label layer"""
        if self.table is not None:
            self.table.hide()
            self.prop_filter_widget.setVisible(False)

        if (
            self.viewer is not None
            and self.label_manager.selected_layer is not None
            and len(self.label_manager.selected_layer.properties) > 0
        ):
            self.table = ColoredTableWidget(
                self.label_manager.selected_layer, self.viewer
            )
            self.table.setMinimumWidth(500)
            self.regionprops_layout.addWidget(self.table)
            self.prop_filter_widget.setVisible(True)

    def get_selected_features(self) -> list[str]:
        """Return a list of the features that have been selected"""

        selected_features = [
            key for key in self.checkbox_state if self.checkbox_state[key]
        ]
        if "intensity_mean" in selected_features and not isinstance(
            self.intensity_image_dropdown.selected_layer, napari.layers.Image
        ):
            selected_features.remove("intensity_mean")
        selected_features.append("label")  # always include label
        selected_features.append("centroid")
        return selected_features

    def set_selected_features(self, features: list[str]) -> None:
        """Set the selected features based on the input list"""

        for checkbox in self.checkboxes:
            checkbox["checkbox"].setChecked(checkbox["region_prop_name"] in features)
