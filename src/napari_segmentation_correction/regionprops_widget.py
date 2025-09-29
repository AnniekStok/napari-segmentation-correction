import dask.array as da
import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
        self.label_manager.layer_update.connect(self.update_dims)
        self.table = None
        self.ndims = 2
        self.feature_dims = 2
        self.axis_widgets = []

        dim_box = QGroupBox("Dimensions")
        grid = QGridLayout()

        # headers
        grid.addWidget(QLabel("Axis"), 0, 0)
        grid.addWidget(QLabel("Index"), 0, 1)
        grid.addWidget(QLabel("Pixel scaling"), 0, 2)

        # Z row
        self.z_label = QLabel("Z")
        self.z_label.setVisible(False)
        self.z_axis = QLabel("0")
        self.z_axis.setVisible(False)
        self.axis_widgets.append(self.z_axis)
        self.z_scale = QDoubleSpinBox()
        self.z_scale.setValue(1.0)
        self.z_scale.setSingleStep(0.1)
        self.z_scale.setMinimum(0.01)
        self.z_scale.setToolTip("Voxel size along Z axis")
        self.z_scale.setVisible(False)
        self.z_scale.setDecimals(3)

        grid.addWidget(self.z_label, 1, 0)
        grid.addWidget(self.z_axis, 1, 1)
        grid.addWidget(self.z_scale, 1, 2)

        # Y row
        self.y_label = QLabel("Y")
        self.y_axis = QLabel("1")
        self.axis_widgets.append(self.y_axis)
        self.y_scale = QDoubleSpinBox()
        self.y_scale.setValue(1.0)
        self.y_scale.setSingleStep(0.1)
        self.y_scale.setMinimum(0.01)
        self.y_scale.setToolTip("Voxel size along Y axis")
        self.y_scale.setDecimals(3)

        grid.addWidget(self.y_label, 2, 0)
        grid.addWidget(self.y_axis, 2, 1)
        grid.addWidget(self.y_scale, 2, 2)

        # X row
        self.x_label = QLabel("X")
        self.x_axis = QLabel("2")
        self.axis_widgets.append(self.x_axis)
        self.x_scale = QDoubleSpinBox()
        self.x_scale.setValue(1.0)
        self.x_scale.setSingleStep(0.1)
        self.x_scale.setMinimum(0.01)
        self.x_scale.setToolTip("Voxel size along X axis")
        self.x_scale.setDecimals(3)

        grid.addWidget(self.x_label, 3, 0)
        grid.addWidget(self.x_axis, 3, 1)
        grid.addWidget(self.x_scale, 3, 2)

        # add "use z" checkbox above grid
        main_layout = QVBoxLayout()
        self.use_z = QCheckBox("3D data (use Z axis)")
        self.use_z.setVisible(False)
        self.use_z.setEnabled(False)
        self.use_z.stateChanged.connect(self.update_use_z)
        main_layout.addWidget(self.use_z)
        main_layout.addLayout(grid)

        dim_box.setLayout(main_layout)
        dim_box.setMaximumHeight(200)

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

        ### Add widget for property filtering
        self.prop_filter_widget = PropertyFilterWidget(self.viewer, self.label_manager)
        self.prop_filter_widget.setVisible(False)

        # Add table layout
        self.regionprops_layout = QVBoxLayout()

        # Assemble layout
        main_box = QGroupBox("Region properties")
        main_layout = QVBoxLayout()
        main_layout.addWidget(dim_box)
        main_layout.addWidget(feature_box)
        main_layout.addWidget(self.measure_btn)
        main_layout.addWidget(self.prop_filter_widget)
        main_layout.addLayout(self.regionprops_layout)
        main_box.setLayout(main_layout)

        layout = QVBoxLayout()
        layout.addWidget(main_box)

        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)

        # refresh dimensions based on current label layer, if any
        self.update_dims()

    def update_properties(self) -> None:
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

    def update_use_z(self, state: int) -> None:
        self.z_label.setVisible(state == 2)
        self.z_axis.setVisible(state == 2)
        self.z_scale.setVisible(state == 2)
        self.z_axis.setEnabled(state == 2)
        self.z_scale.setEnabled(state == 2)
        self.feature_dims = 3 if state == 2 else 2
        self.update_dims()

    def update_dims(self) -> None:
        """Update the number of dimensions to measure based on the selected checkboxes"""

        if self.label_manager.selected_layer is not None:
            self.measure_btn.setEnabled(True)
            self.ndims = self.label_manager.selected_layer.ndim
            self.use_z.setVisible(self.ndims == 3)
            self.use_z.setEnabled(self.ndims == 3)
            if self.ndims == 4:
                self.feature_dims = 3
                self.z_axis.setVisible(True)
                self.z_label.setVisible(True)
                self.z_scale.setVisible(True)
                self.z_axis.setEnabled(True)
                self.z_scale.setEnabled(True)

            ax_names = [str(ax) for ax in range(self.ndims)]
            if len(ax_names) > 0:
                for i, widget in enumerate(self.axis_widgets):
                    if self.ndims == 4:
                        widget.setText(ax_names[i + 1])
                    elif self.ndims == 2:
                        widget.setText(ax_names[i - 1])
                    else:
                        widget.setText(ax_names[i])

                self.update_properties()
                self.update_table()

            self.z_scale.setValue(self.label_manager.selected_layer.scale[-3]) if len(
                self.label_manager.selected_layer.scale
            ) >= 3 else self.z_scale.setValue(1.0)
            self.y_scale.setValue(self.label_manager.selected_layer.scale[-2])
            self.x_scale.setValue(self.label_manager.selected_layer.scale[-1])

        else:
            self.measure_btn.setEnabled(False)

    def measure(self):
        if self.use_z.isChecked() or self.ndims == 4:
            spacing = (self.z_scale.value(), self.y_scale.value(), self.x_scale.value())
        else:
            spacing = (self.y_scale.value(), self.x_scale.value())

        # ensure spacing is applied to the layer and the viewer step is updated
        layer_scale = list(self.label_manager.selected_layer.scale)
        layer_scale[-1] = spacing[-1]
        layer_scale[-2] = spacing[-2]
        if len(layer_scale) > 3:
            layer_scale[-3] = spacing[-3]
        old_step = list(self.viewer.dims.current_step)
        step_size = [dim_range.step for dim_range in self.viewer.dims.range]
        new_step = [
            step * step_size
            for step, step_size in zip(old_step, step_size, strict=False)
        ]
        self.label_manager.selected_layer.scale = layer_scale
        self.viewer.reset_view()
        self.viewer.dims.current_step = new_step

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
        if (self.use_z.isChecked() and self.ndims == 3) or (
            not self.use_z.isChecked() and self.ndims == 2
        ):
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
