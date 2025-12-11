import dask.array as da
import napari
import numpy as np
import pandas as pd
from napari.utils import CyclicLabelColormap, DirectLabelColormap
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

from napari_segmentation_toolbox.helpers.base_tool_widget import BaseToolWidget
from napari_segmentation_toolbox.helpers.layer_dropdown import LayerDropdown
from napari_segmentation_toolbox.regionprops.color_feature_widget import (
    ColorFeatureWidget,
)
from napari_segmentation_toolbox.regionprops.custom_table_widget import (
    ColoredTableWidget,
)
from napari_segmentation_toolbox.regionprops.prop_filter_widget import (
    PropertyFilterWidget,
)
from napari_segmentation_toolbox.regionprops.regionprops_extended import (
    calculate_extended_props,
)


def reorder_array(A_axes, B_axes, B_array):
    """
    Reorder B_array so its axes match A_axes as much as possible.
    Then move 'C' to the last axis if present.

    Parameters
    ----------
    A_axes : list[str]
        Axis order of A, e.g. ['Z','T','Y','X']
    B_axes : list[str]
        Axis order of B, e.g. ['C','T','Z','Y','X']
    B_array : np.ndarray
        Array to reorder.

    Returns
    -------
    B_reordered : np.ndarray
    new_axes : list[str]
        New axis ordering after reordering + moving C to end.
    """

    common = [ax for ax in A_axes if ax in B_axes]
    remaining = [ax for ax in B_axes if ax not in A_axes]
    new_axes = remaining + common

    # Map this order to B's current axes
    transpose_order = [B_axes.index(ax) for ax in new_axes]
    B_reordered = np.transpose(B_array, transpose_order)

    if "C" in new_axes:
        c_pos = new_axes.index("C")
        if c_pos != len(new_axes) - 1:
            # Move C to the end
            new_axes.append(new_axes.pop(c_pos))
            B_reordered = np.moveaxis(B_reordered, c_pos, -1)

    return B_reordered, new_axes


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


def slice_axis(arr, index, axis):
    """Safely slice along a given axis for numpy or dask arrays."""
    if isinstance(arr, da.core.Array):
        return da.take(arr, index, axis=axis)
    else:
        return np.take(arr, index, axis=axis)


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

                append = "Control" in event.modifiers
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
        """Measure selected region properties and update table, looping over C and T if present."""

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

        if intensity_layer is not None and "dimensions" not in intensity_layer.metadata:
            msg = QMessageBox()
            msg.setWindowTitle("Missing metadata")
            msg.setText(
                "Intensity layer does not have dimensions metadata. Please activate (select) the layer and ensure the dimensions are listed correctly to ensure the metadata is populated."
            )
            msg.setIcon(QMessageBox.Critical)
            msg.exec_()
            return

        intensity_dims = intensity_layer.metadata["dimensions"]

        # identify axes
        has_time = "T" in dims
        has_channel = "C" in dims
        time_axis = dims.index("T") if has_time else None
        t_range = range(data.shape[time_axis]) if has_time else [None]

        channel_axis = dims.index("C") if has_channel else None
        c_range = range(data.shape[channel_axis]) if has_channel else [None]

        if (
            time_axis is not None
            and channel_axis is not None
            and channel_axis > time_axis
        ):
            channel_axis -= 1  # because we slice time first

        if intensity is not None:
            intensity, intensity_dims = reorder_array(dims, intensity_dims, intensity)

        # prepare for nested loops over T and C
        prop_list = []

        try:
            for t in tqdm(t_range, desc="Time"):
                data_t = slice_axis(data, t, time_axis) if has_time else data

                for c in tqdm(c_range, desc="Channel", leave=False):
                    data_tc = (
                        slice_axis(data_t, c, channel_axis) if has_channel else data_t
                    )

                    # slice intensity if present
                    if intensity is not None:
                        int_slice = intensity
                        if has_time:
                            int_t_dim = intensity_dims.index("T")
                            int_slice = slice_axis(int_slice, t, int_t_dim)
                    else:
                        int_slice = None

                    p = calculate_extended_props(
                        data_tc,
                        intensity_image=int_slice,
                        properties=features,
                        spacing=spatial_spacing,
                    )

                    if has_time:
                        p["time_point"] = t
                    if has_channel:
                        p["channel"] = c

                    prop_list.append(p)

        except ValueError as e:
            msg = QMessageBox()
            msg.setWindowTitle("Error in measuring region properties")
            msg.setText(
                f"Region properties could not be computed: {e}. Please check if your labels layer and intensity image layer are a valid match."
            )
            msg.setIcon(QMessageBox.Critical)
            msg.exec_()
            return

        # concatenate all properties
        props = pd.concat(prop_list, ignore_index=True)

        # update layer properties
        self.layer.properties = props
        self.prop_filter_widget.set_properties()
        self.color_by_feature_widget.set_properties()
        self._convert_layer_colormap()
        self._update_table()

    def _convert_layer_colormap(self):
        """replace cyclic map by direct map if necessary"""

        if isinstance(self.layer.colormap, CyclicLabelColormap):
            labels = self.layer.properties["label"]
            colors = [self.layer.colormap.map(label) for label in labels]
            self.layer.colormap = DirectLabelColormap(
                color_dict={
                    **dict(zip(labels, colors, strict=True)),
                    None: [0, 0, 0, 0],
                }
            )

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
