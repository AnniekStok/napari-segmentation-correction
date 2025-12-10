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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from napari_segmentation_correction.helpers.base_tool_widget import BaseToolWidget
from napari_segmentation_correction.helpers.layer_dropdown import LayerDropdown
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


def slice_axis(arr: np.ndarray | da.core.Array, idx: int, axis: int) -> np.ndarray:
    """Returns arr sliced at position idx along the given axis.
    Works for both NumPy and Dask arrays."""

    if isinstance(arr, da.core.Array):
        sl = [slice(None)] * arr.ndim
        sl[axis] = idx
        return arr[tuple(sl)].compute()
    else:
        sl = [slice(None)] * arr.ndim
        sl[axis] = idx
        return arr[tuple(sl)]


def reorder_intensity_like_labels(
    labels: np.ndarray, intensity: np.ndarray
) -> np.ndarray | None:
    """Reorder intensity array:
        - intensity must match labels, or
        - intensity may have an extra trailing C dimension.
    Reorder axes of `intensity` so its leading axes match `labels.shape`.
    If intensity has one extra axis, that axis will be moved to the end of the returned array.
    Raises ValueError if matching is impossible.
    """
    if intensity is None:
        return None
    if intensity.shape == labels.shape:
        return intensity

    shape_labels = labels.shape
    shape_intensity = intensity.shape
    nd_labels = len(shape_labels)
    nd_intensity = len(shape_intensity)

    # same number of dims: try to permute intensity to match labels
    if nd_intensity == nd_labels:
        for perm in permutations(range(nd_intensity)):
            if tuple(shape_intensity[i] for i in perm) == shape_labels:
                return intensity.transpose(perm)
        raise ValueError(
            f"No permutation of b.shape {shape_intensity} matches a.shape {shape_labels}."
        )

    # intensity has exactly one extra dim: try every axis as the extra one
    if nd_intensity == nd_labels + 1:
        axes = list(range(nd_intensity))
        for extra_axis in axes:
            remaining_axes = [ax for ax in axes if ax != extra_axis]
            # try permutations of the remaining axes to match shape_labels
            for perm_remaining in permutations(remaining_axes):
                if tuple(shape_intensity[i] for i in perm_remaining) == shape_labels:
                    # final permutation: place the permuted remaining axes first, then the extra axis last
                    final_perm = list(perm_remaining) + [extra_axis]
                    return intensity.transpose(final_perm)
        raise ValueError(
            f"Intensity image has one extra axis but no permutation places its other axes in order {shape_labels} "
            f"with the extra axis last. intensity shape={shape_intensity}, labels shape={shape_labels}"
        )

    # any other difference in rank is invalid
    raise ValueError(
        "Intensity image must have either the same number of dimensions as labels, or exactly one more."
        f" Got intensity ndim={nd_labels}, labels ndim={nd_intensity}."
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


class RegionPropsWidget(BaseToolWidget):
    """Widget to compute region properties, display a table, and allow the user to color
    labels by feature and to filter by feature.
    """

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(napari.layers.Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        self.tab_widget = QTabWidget(self)
        self.table = None
        self._source_callback = None

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
        self.prop_filter_widget = PropertyFilterWidget(self.viewer)
        self.prop_filter_widget.setVisible(False)

        ### Add widget to color by feature
        self.color_by_feature_widget = ColorFeatureWidget(self.viewer)
        self.color_by_feature_widget.setVisible(False)

        # Add table layout
        regionprops_layout = QVBoxLayout()

        # Assemble layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(intensity_box)
        main_layout.addWidget(shape_box)
        main_layout.addWidget(self.measure_btn)
        main_layout.addWidget(self.prop_filter_widget)
        main_layout.addWidget(self.color_by_feature_widget)
        main_layout.addLayout(regionprops_layout)
        main_layout.setAlignment(Qt.AlignTop)
        settings_widget = QWidget()
        settings_widget.setLayout(main_layout)

        table_widget = QWidget()
        self.table_layout = QVBoxLayout()
        table_widget.setLayout(self.table_layout)

        self.tab_widget.addTab(settings_widget, "Settings")
        self.tab_widget.addTab(table_widget, "Table")
        self.tab_widget.setCurrentIndex(0)

        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)

        self.setLayout(layout)

        # connect to update signal
        self.update_status.connect(self.update_properties_and_callback)
        self.update_properties_and_callback()

    def _update_measure_btn_state(self, state: int | None = None) -> None:
        """Update the button state according to whether at least one checkbox is checked"""

        if self.layer is None:
            self.measure_btn.setEnabled(False)
            return

        checked = [
            ch["region_prop_name"]
            for ch in self.checkboxes
            if (ch["checkbox"].isChecked() and ch["checkbox"].isEnabled())
        ]

        self.measure_btn.setEnabled(True) if len(
            checked
        ) > 0 else self.measure_btn.setEnabled(False)

    def _table_callback(self, layer) -> callable:
        """Create a callback function for copying labels from the source layer to"""

        def callback(layer, event):
            if (
                event.type == "mouse_press"
                and self.table is not None
                and self.table.isVisible()
            ):
                selected_label = layer.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True,
                )

                append = "Shift" in event.modifiers
                self.table.select_label(event.position, selected_label, append=append)

        return callback

    def update_properties_and_callback(self) -> None:
        """Update the available properties based on the selected label layer dimensions"""

        if self.layer is not None and self._source_callback is not None:
            try:
                self.layer.mouse_drag_callbacks.remove(self._source_callback)
                self.layer.contour = 0
            except ValueError:
                pass

        if self.layer is not None:
            self._source_callback = self._table_callback(self.layer)
            self.layer.mouse_drag_callbacks.append(self._source_callback)

        if self.layer is not None and "dimensions" in self.layer.metadata:
            dims = self.layer.metadata["dimensions"]
            feature_dims = 3 if "Z" in dims else 2
        else:
            feature_dims = 2

        # Set the visibility of each checkbox according to the dimensions
        visible_props = [
            prop["region_prop_name"]
            for prop in intensity_properties + shape_properties
            if (feature_dims in prop["dims"] and self.layer is not None)
        ]
        for checkbox_dict in self.checkboxes:
            prop = checkbox_dict["region_prop_name"]
            visible = prop in visible_props
            checkbox_dict["checkbox"].setVisible(visible)
            checkbox_dict["checkbox"].setEnabled(visible)

        self.intensity_image_dropdown.setVisible(self.layer is not None)

        if hasattr(self.layer, "properties"):
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

    def _measure(self) -> None:
        """Measure selected region properties and update table."""

        def drop_axis(arr, axis):
            if isinstance(arr, da.core.Array):
                return da.take(arr, self.viewer.dims.current_step[axis], axis=axis)
            else:
                return np.take(arr, self.viewer.dims.current_step[axis], axis=axis)

        # extract dims and spacing
        dims = list(self.layer.metadata["dimensions"])
        data = self.layer.data
        spacing = self.layer.scale

        # Remove spacing entries for C/T so spacing aligns with spatial dims
        spatial_spacing = [
            s for s, d in zip(spacing, dims, strict=False) if d not in ("C", "T")
        ]

        # selected features
        features = self._get_selected_features()

        # intensity image
        intensity_layer = self.intensity_image_dropdown.selected_layer
        if any(f.startswith("intensity") for f in features) and isinstance(
            intensity_layer, napari.layers.Image | napari.layers.Labels
        ):
            intensity = intensity_layer.data
        else:
            intensity = None

        # remove C dimension (labels must NOT have C)
        if "C" in dims:
            c_axis = dims.index("C")
            data = drop_axis(data, c_axis)
            dims.pop(c_axis)

        # reorder intensity if needed (C must be last for regionprops)
        if intensity is not None:
            try:
                intensity = reorder_intensity_like_labels(data, intensity)
            except ValueError:
                msg = QMessageBox()
                msg.setWindowTitle("Shape mismatch")
                msg.setText(
                    f"Label layer and intensity image must have compatible shapes.\n"
                    f"Labels: {data.shape}\nIntensity: {intensity.shape}"
                )
                msg.setIcon(QMessageBox.Critical)
                msg.exec_()
                return

        # choose 2D or 3D props
        is_time = "T" in dims
        time_axis = dims.index("T") if is_time else None

        # handle non-time case
        if not is_time:
            props = calculate_extended_props(
                data,
                intensity_image=intensity,
                properties=features,
                spacing=spatial_spacing,
            )
        else:
            # iterate over T
            nT = data.shape[time_axis]
            prop_list = []

            for t in tqdm(range(nT)):
                lbl = slice_axis(data, t, time_axis)
                if intensity is not None and intensity.ndim > lbl.ndim:
                    # intensity has trailing C → slice T but not C
                    int_slice = slice_axis(intensity, t, time_axis)
                elif intensity is not None:
                    int_slice = slice_axis(intensity, t, time_axis)
                else:
                    int_slice = None

                p = calculate_extended_props(
                    lbl,
                    intensity_image=int_slice,
                    properties=features,
                    spacing=spatial_spacing,
                )
                p["time_point"] = t
                prop_list.append(p)

            props = pd.concat(prop_list, ignore_index=True)

        # update layer properties
        self.layer.properties = props
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
            and self.layer is not None
            and len(self.layer.properties) > 0
        ):
            self.table = ColoredTableWidget(self.layer, self.viewer)
            self.table.setMinimumWidth(500)
            self.table_layout.addWidget(self.table)
            self.prop_filter_widget.setVisible(True)
            self.color_by_feature_widget.setVisible(True)
            self.tab_widget.setCurrentIndex(1)
