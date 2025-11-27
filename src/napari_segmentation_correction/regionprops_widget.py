from itertools import permutations

import dask.array as da
import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from napari_segmentation_correction.helpers.layer_dropdown import LayerDropdown
from napari_segmentation_correction.layer_control_widgets.layer_manager import (
    LayerManager,
)
from napari_segmentation_correction.regionprops.color_feature_widget import (
    ColorFeatureWidget,
)
from napari_segmentation_correction.regionprops.custom_table_widget import (
    ColoredTableWidget,
)
from napari_segmentation_correction.regionprops.prop_filter_widget import (
    PropertyFilterWidget,
)
from napari_segmentation_correction.regionprops.regionprops_extended import (
    calculate_extended_props,
)


def reorder_to_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Reorder axes of `b` so its leading axes match `a.shape`.
    If b has one extra axis, that axis will be moved to the end of the returned array.
    Raises ValueError if matching is impossible.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    shape_a = a.shape
    shape_b = b.shape
    nd_a = len(shape_a)
    nd_b = len(shape_b)

    # same number of dims: try to permute b to match a
    if nd_b == nd_a:
        for perm in permutations(range(nd_b)):
            if tuple(shape_b[i] for i in perm) == shape_a:
                return b.transpose(perm)
        raise ValueError(
            f"No permutation of b.shape {shape_b} matches a.shape {shape_a}."
        )

    # b has exactly one extra dim: try every axis as the extra one
    if nd_b == nd_a + 1:
        axes = list(range(nd_b))
        for extra_axis in axes:
            remaining_axes = [ax for ax in axes if ax != extra_axis]
            # try permutations of the remaining axes to match shape_a
            for perm_remaining in permutations(remaining_axes):
                if tuple(shape_b[i] for i in perm_remaining) == shape_a:
                    # final permutation: place the permuted remaining axes first, then the extra axis last
                    final_perm = list(perm_remaining) + [extra_axis]
                    return b.transpose(final_perm)
        raise ValueError(
            f"b has one extra axis but no permutation places its other axes in order {shape_a} "
            f"with the extra axis last. b.shape={shape_b}, a.shape={shape_a}"
        )

    # any other difference in rank is invalid
    raise ValueError(
        "b must have either the same number of dimensions as a, or exactly one more."
        f" Got a.ndim={nd_a}, b.ndim={nd_b}."
    )


intensity_properties = [
    {
        "region_prop_name": "intensity_mean",
        "display_name": "Mean intensity",
        "enabled": False,
        "dims": [2, 3],
    },
    {
        "region_prop_name": "intensity_min",
        "display_name": "Min intensity",
        "enabled": False,
        "dims": [2, 3],
    },
    {
        "region_prop_name": "intensity_max",
        "display_name": "Max intensity",
        "enabled": False,
        "dims": [2, 3],
    },
]
shape_properties = [
    {
        "region_prop_name": "area",
        "display_name": "Area",
        "enabled": True,
        "dims": [2],
    },
    {
        "region_prop_name": "perimeter",
        "display_name": "Perimeter",
        "enabled": True,
        "dims": [2],
    },
    {
        "region_prop_name": "circularity",
        "display_name": "Circularity",
        "enabled": True,
        "dims": [2],
    },
    {
        "region_prop_name": "ellipse_axes",
        "display_name": "Ellipse axes",
        "enabled": True,
        "dims": [2],
    },
    {
        "region_prop_name": "volume",
        "display_name": "Volume",
        "enabled": True,
        "dims": [3],
    },
    {
        "region_prop_name": "surface_area",
        "display_name": "Surface area",
        "enabled": True,
        "dims": [3],
    },
    {
        "region_prop_name": "sphericity",
        "display_name": "Sphericity",
        "enabled": True,
        "dims": [3],
    },
    {
        "region_prop_name": "ellipsoid_axes",
        "display_name": "Ellipsoid axes",
        "enabled": True,
        "dims": [3],
    },
]


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

        intensity_box = QGroupBox("Intensity features")
        intensity_box.setMaximumHeight(140)
        shape_box = QGroupBox("Shape features")
        shape_box.setMaximumHeight(130)

        self.intensity_checkbox_layout = QVBoxLayout()
        self.shape_checkbox_layout = QVBoxLayout()
        self.checkboxes = []

        self.intensity_image_dropdown = LayerDropdown(
            self.viewer, (napari.layers.Image, napari.layers.Labels)
        )
        self.intensity_checkbox_layout.addWidget(self.intensity_image_dropdown)

        for prop in intensity_properties:
            checkbox = QCheckBox(prop["display_name"])
            checkbox.setEnabled(prop["enabled"])
            checkbox.setStyleSheet("QCheckBox:disabled { color: grey }")
            checkbox.stateChanged.connect(self._update_measure_btn_state)

            self.checkboxes.append(
                {"region_prop_name": prop["region_prop_name"], "checkbox": checkbox}
            )
            self.intensity_checkbox_layout.addWidget(checkbox)

        intensity_box.setLayout(self.intensity_checkbox_layout)

        for prop in shape_properties:
            checkbox = QCheckBox(prop["display_name"])
            checkbox.setEnabled(prop["enabled"])
            checkbox.setStyleSheet("QCheckBox:disabled { color: grey }")
            checkbox.stateChanged.connect(self._update_measure_btn_state)

            self.checkboxes.append(
                {"region_prop_name": prop["region_prop_name"], "checkbox": checkbox}
            )
            self.shape_checkbox_layout.addWidget(checkbox)

        shape_box.setLayout(self.shape_checkbox_layout)

        # Push button to measure features
        self.measure_btn = QPushButton("Measure properties")
        self.measure_btn.clicked.connect(self._measure)
        self.measure_btn.setEnabled(False)

        ### Add widget for property filtering
        self.prop_filter_widget = PropertyFilterWidget(self.viewer, self.label_manager)
        self.prop_filter_widget.setVisible(False)

        ### Add widget to color by feature
        self.color_by_feature_widget = ColorFeatureWidget(
            self.viewer, self.label_manager
        )
        self.color_by_feature_widget.setVisible(False)

        # Add table layout
        self.regionprops_layout = QVBoxLayout()

        # Assemble layout
        main_box = QGroupBox("Region properties")
        main_layout = QVBoxLayout()
        main_layout.addWidget(intensity_box)
        main_layout.addWidget(shape_box)
        main_layout.addWidget(self.measure_btn)
        main_layout.addWidget(self.prop_filter_widget)
        main_layout.addWidget(self.color_by_feature_widget)
        main_layout.addLayout(self.regionprops_layout)
        main_box.setLayout(main_layout)

        layout = QVBoxLayout()
        layout.addWidget(main_box)

        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)

        # connect to update signal
        self.label_manager.layer_update.connect(self._update_properties)
        self._update_properties()

    def _update_measure_btn_state(self, state: int | None = None):
        """Update the button state according to whether at least one checkbox is checked"""

        checked = [
            ch["region_prop_name"]
            for ch in self.checkboxes
            if (ch["checkbox"].isChecked() and ch["checkbox"].isEnabled())
        ]

        self.measure_btn.setEnabled(True) if len(
            checked
        ) > 0 else self.measure_btn.setEnabled(False)

    def _update_properties(self) -> None:
        """Update the available properties based on the selected label layer dimensions"""

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

        # Set the visibility of each checkbox according to the dimensions
        visible_props = [
            prop["region_prop_name"]
            for prop in intensity_properties + shape_properties
            if (
                self.feature_dims in prop["dims"]
                and self.label_manager.selected_layer is not None
            )
        ]
        for checkbox_dict in self.checkboxes:
            prop = checkbox_dict["region_prop_name"]
            visible = prop in visible_props
            checkbox_dict["checkbox"].setVisible(visible)
            checkbox_dict["checkbox"].setEnabled(visible)

        self.intensity_image_dropdown.setVisible(
            self.label_manager.selected_layer is not None
        )

        if hasattr(self.label_manager.selected_layer, "properties"):
            self._update_table()

        self._update_measure_btn_state()

    def _get_selected_features(self) -> list[str]:
        """Return a list of the features that have been selected"""

        selected_features = [
            ch["region_prop_name"]
            for ch in self.checkboxes
            if (ch["checkbox"].isChecked() and ch["checkbox"].isEnabled())
        ]
        selected_features.append("label")  # always include label
        selected_features.append("centroid")
        return selected_features

    def _measure(self):
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
        features = self._get_selected_features()

        if "intensity_mean" in features and isinstance(
            self.intensity_image_dropdown.selected_layer,
            napari.layers.Image | napari.layers.Labels,
        ):
            intensity_image = self.intensity_image_dropdown.selected_layer.data
        else:
            intensity_image = None

        data = self.label_manager.selected_layer.data
        if intensity_image is not None and intensity_image.shape != data.shape:
            # if shapes don't match, try to transpose the data such that the order is
            # correct. Multichannel intensity images are allowed as long as the channel
            # dimension is the last dimension. Since this does not match with the default
            # napari order of dims, transpose it here.

            try:
                intensity_image = reorder_to_match(data, intensity_image)
            except ValueError:
                msg = QMessageBox()
                msg.setWindowTitle("Shape mismatch")
                msg.setText(
                    f"Label layer and intensity image must have compatible shapees. Got {self.label_manager.selected_layer.data.shape} and {intensity_image.shape}."
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
            self.color_by_feature_widget.set_properties()
            self._update_table()

    def _update_table(self) -> None:
        """Update the regionprops table based on the selected label layer"""
        if self.table is not None:
            self.table.hide()
            self.prop_filter_widget.setVisible(False)
            self.color_by_feature_widget.setVisible(False)
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
            self.color_by_feature_widget.setVisible(True)
