import dask.array as da
import napari
import numpy as np
from napari.layers import Labels
from napari.utils.events import Event
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from napari_segmentation_correction.helpers.layer_dropdown import LayerDropdown


def check_value_dtype(value, dtype: np.dtype) -> tuple[bool, np.dtype]:
    """Check if a given value fits in a given dtype, if not, return the next available
    dtype."""

    # Get min and max for the dtype
    info = np.iinfo(dtype)
    within_range = info.min <= value <= info.max

    # If not in range, find the next suitable unsigned dtype
    next_dtype = None
    if not within_range:
        unsigned_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
        for dt in unsigned_dtypes:
            if np.iinfo(dt).max >= value:
                next_dtype = dt
                break

    return within_range, next_dtype


class DimsRadioButtons(QWidget):
    """Radio buttons to choose dimensions."""

    def __init__(self) -> None:
        super().__init__()

        label = QLabel("Copy dimensions:")

        button_group = QButtonGroup()
        self.slice = QRadioButton("Slice (last 2 dims)")
        self.slice.setEnabled(False)
        self.slice.setChecked(False)
        self.volume = QRadioButton("Volume (last 3 dims)")
        self.volume.setChecked(False)
        self.volume.setEnabled(False)
        self.series = QRadioButton("Series (last 4 dims)")
        self.series.setChecked(False)
        self.series.setEnabled(False)

        button_group.addButton(self.slice)
        button_group.addButton(self.volume)
        button_group.addButton(self.series)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.slice)
        button_layout.addWidget(self.volume)
        button_layout.addWidget(self.series)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addLayout(button_layout)
        self.setLayout(layout)


class CopyLabelWidget(QWidget):
    """Widget to copy labels from a source layer to a target layer."""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer

        self.source_layer = None
        self.target_layer = None
        self._source_callback = None

        copy_labels_box = QGroupBox("Copy-paste labels")
        copy_labels_layout = QVBoxLayout()

        # instruction label
        label = QLabel(
            "Use shift + click on the source layer to copy labels to the target layer."
        )
        label.setWordWrap(True)
        font = label.font()
        font.setItalic(True)
        label.setFont(font)
        copy_labels_layout.addWidget(label)

        # Whether or not to preserve the source layer label value when copying or to use
        # the next available label in the target layer
        self.preserve_label_value = QCheckBox("Use source label value")
        self.preserve_existing_labels = QCheckBox("Preserve target labels")
        option_layout = QHBoxLayout()
        option_layout.addWidget(self.preserve_label_value)
        option_layout.addWidget(self.preserve_existing_labels)
        copy_labels_layout.addLayout(option_layout)

        # Source layer and target layer dropdowns
        image_layout = QVBoxLayout()
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source labels"))
        self.source_dropdown = LayerDropdown(self.viewer, (Labels), allow_none=True)
        self.source_dropdown.viewer.layers.selection.events.changed.disconnect(
            self.source_dropdown._on_selection_changed
        )
        self.source_dropdown.layer_changed.connect(self._update_source)
        source_layout.addWidget(self.source_dropdown)

        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target labels"))
        self.target_dropdown = LayerDropdown(self.viewer, (Labels), allow_none=True)
        self.target_dropdown.viewer.layers.selection.events.changed.disconnect(
            self.target_dropdown._on_selection_changed
        )
        self.target_dropdown.layer_changed.connect(self._update_target)
        target_layout.addWidget(self.target_dropdown)

        image_layout.addLayout(source_layout)
        image_layout.addLayout(target_layout)

        copy_labels_layout.addLayout(image_layout)

        # Radiobuttons for selecting whether to copy a slice/volume/series
        self.dims_widget = DimsRadioButtons()
        copy_labels_layout.addWidget(self.dims_widget)

        # Undo the last copy action if possible
        self.prev_state = None
        self.coords_clipped = None
        self.target_slices = None
        self.undo_btn = QPushButton("Undo last copy")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo)
        copy_labels_layout.addWidget(self.undo_btn)

        # assemble the layout
        copy_labels_box.setLayout(copy_labels_layout)
        layout = QVBoxLayout()
        layout.addWidget(copy_labels_box)
        self.setLayout(layout)

    def _update_source(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying
        labels from."""

        if self.source_layer is not None and self._source_callback is not None:
            try:
                self.source_layer.mouse_drag_callbacks.remove(self._source_callback)
                self.source_layer.contour = 0
            except ValueError:
                pass
        if selected_layer == "":
            self.source_layer = None
            self._source_callback = None
        else:
            self.source_layer = self.viewer.layers[selected_layer]
            self.source_layer.contour = 1
            self.source_dropdown.setCurrentText(selected_layer)
            self._source_callback = self._make_copy_label_callback(self.source_layer)
            self.source_layer.mouse_drag_callbacks.append(self._source_callback)

        self.update_radiobuttons()

    def _update_target(self, selected_layer: str) -> None:
        """Update the layer to copy labels to."""

        if selected_layer == "":
            self.target_layer = None
        else:
            self.target_layer = self.viewer.layers[selected_layer]
            self.target_dropdown.setCurrentText(selected_layer)

        self.update_radiobuttons()

    def update_radiobuttons(self) -> None:
        """Update the state of the dimension checkboxes based on the source and target
        layers."""

        # All buttons off
        self.dims_widget.slice.setEnabled(False)
        self.dims_widget.slice.setChecked(False)
        self.dims_widget.volume.setEnabled(False)
        self.dims_widget.volume.setChecked(False)
        self.dims_widget.series.setEnabled(False)
        self.dims_widget.series.setChecked(False)
        self.undo_btn.setEnabled(False)

        if self.source_layer is not None and self.target_layer is not None:
            # Set the highest possible option based on the number of dimensions of the
            # source and target layers
            source_dims = self.source_layer.data.ndim
            target_dims = self.target_layer.data.ndim
            dims = min(source_dims, target_dims)

            if dims >= 4:
                self.dims_widget.series.setEnabled(True)
                self.dims_widget.volume.setEnabled(True)
                self.dims_widget.slice.setEnabled(True)
                self.dims_widget.volume.setChecked(True)
            elif dims >= 3:
                self.dims_widget.volume.setEnabled(True)
                self.dims_widget.slice.setEnabled(True)
                self.dims_widget.volume.setChecked(True)
            elif dims >= 2:
                self.dims_widget.slice.setEnabled(True)
                self.dims_widget.slice.setChecked(True)

    def _make_copy_label_callback(self, layer: Labels) -> callable:
        """Create a callback function for copying labels from the source layer to"""

        def callback(layer, event):
            if event.type == "mouse_press" and "Shift" in event.modifiers:
                self.copy_label(event)

        return callback

    def copy_label(self, event: Event, source_layer: Labels | None = None) -> None:
        """Copy a 2D/3D/4D label from this layer to a target layer"""

        if self.source_layer is None or self.target_layer is None:
            return

        # Check whether to copy a slice/volume/series label according to the
        # radiobutton choice
        if self.dims_widget.series.isChecked():
            n_dims_copied = 4
        elif self.dims_widget.volume.isChecked():
            n_dims_copied = 3
        else:
            n_dims_copied = 2

        if n_dims_copied == 4 and (
            isinstance(self.source_layer.data, da.core.Array)
            or isinstance(self.target_layer.data, da.core.Array)
        ):
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText(
                "Copying labels in 4D dimensions between dask arrays is slow, are you sure you want to continue?"
            )
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            result = msg.exec_()
            if result != QMessageBox.Ok:
                return

        # extract label value from source layer (orthoview or self.source_layer)
        source_layer = source_layer if source_layer is not None else self.source_layer
        selected_label = source_layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )

        # do not process clicking on the background
        if selected_label == 0:
            return

        # extract coords from click position
        coords = self.source_layer.world_to_data(event.position)
        coords = [int(c) for c in coords]

        # Get dimensions of source and target layers
        dims_displayed = event.dims_displayed
        ndims_source = len(self.source_layer.data.shape)
        ndims_target = len(self.target_layer.data.shape)

        # Assign a new label value if None is provided
        target_label = selected_label if self.preserve_label_value.isChecked() else None
        if target_label is None:
            target_label = np.max(self.target_layer.data) + 1

        # Check if the target label is within the dtype range of the target layer, if not
        # suggest converting to a larger dtype
        within_range, next_dtype = check_value_dtype(
            target_label, self.target_layer.data.dtype
        )
        if not within_range:
            msg = QMessageBox()
            msg.setWindowTitle("Invalid label!")
            if next_dtype is not None:
                msg.setText(
                    f"Label {target_label} exceeds bit depth! Convert to {next_dtype}?"
                )
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                result = msg.exec_()
                if result == QMessageBox.Ok:  # Check if Ok was clicked
                    self.target_layer.data = self.target_layer.data.astype(next_dtype)
                else:
                    return

        # Select dims to copy
        source_shape = self.source_layer.data.shape
        labels_shape = self.target_layer.data.shape
        source_dims_to_copy = source_shape[-n_dims_copied:]
        target_dims_to_copy = labels_shape[-n_dims_copied:]

        # Check if the dimensions to copy match
        if source_dims_to_copy != target_dims_to_copy:
            msg = QMessageBox()
            msg.setWindowTitle("Invalid dimensions!")
            msg.setText(
                f"The dimensions of the source layer and the target layer do not match.\n"
                f"Label source layer has shape {source_shape}, target layer has shape {labels_shape}.\n"
                f"Compared last {n_dims_copied} dims: {source_dims_to_copy} vs {target_dims_to_copy}."
            )
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        # Create source_slices for all dimensions of the source layer
        source_slices = [slice(None)] * ndims_source

        # When copying 2D labels, we need to check the dimensions displayed, in case
        # the sure is copying from one of the orthoviews.
        dims_difference = ndims_source - ndims_target
        if n_dims_copied == 2:
            # Create a list of `slice(None)` for all dimensions of self.source_layer.data
            if dims_difference < 0:
                dims_displayed = [dim + dims_difference for dim in dims_displayed]
            for i in range(ndims_source):
                if i not in dims_displayed:
                    source_slices[i] = coords[
                        i
                    ]  # Replace the slice with a specific coordinate for slider dims

        else:
            # Calculate the coords for the remaining dims
            remaining_coords = coords[:-n_dims_copied]
            for i, coord in enumerate(remaining_coords):
                source_slices[i] = coord

        # Clip coords to the shape of the target data
        if dims_difference > 0:
            coords_clipped = coords[dims_difference:]
            target_slices = source_slices[dims_difference:]
        elif dims_difference < 0:
            coords_clipped = [
                *self.viewer.dims.current_step[: abs(dims_difference)],
                *coords,
            ]
            target_slices = [
                *self.viewer.dims.current_step[: abs(dims_difference)],
                *source_slices,
            ]
        else:
            coords_clipped = coords
            target_slices = source_slices

        # Create mask
        if isinstance(self.source_layer.data, da.core.Array):
            mask = (
                self.source_layer.data[tuple(source_slices)].compute() == selected_label
            )
        else:
            mask = self.source_layer.data[tuple(source_slices)] == selected_label

        # Select the correct stack for 2D/3D/4D data
        orig_label = self.target_layer.data[tuple(coords_clipped)]
        sliced_data = self.target_layer.data[tuple(target_slices)]
        if isinstance(sliced_data, da.core.Array):
            sliced_data = sliced_data.compute()

        # Store previous state for undo
        self.prev_state = np.copy(sliced_data)
        self.target_slices = np.copy(target_slices)
        self.coords_clipped = np.copy(coords_clipped)

        # Replace label in target layer data
        orig_mask = sliced_data == orig_label
        if not self.preserve_existing_labels.isChecked():
            sliced_data[orig_mask] = 0
            sliced_data[mask] = target_label
        else:
            sliced_data[orig_mask & (sliced_data == 0) & mask] = target_label

        self.target_layer.data[tuple(target_slices)] = sliced_data
        self.undo_btn.setEnabled(True)

        # refresh the layer
        self.target_layer.data = self.target_layer.data

    def undo(self) -> None:
        """Undo the last label copy operation"""

        if hasattr(self, "prev_state") and self.prev_state is not None:
            self.target_layer.data[tuple(self.target_slices)] = self.prev_state

            # refresh the layer
            self.target_layer.data = self.target_layer.data

            # set back to None and disable button
            self.prev_state = None
            self.coords_clipped = None
            self.target_slices = None
            self.undo_btn.setEnabled(False)

    def sync_click(
        self, orig_layer: Labels, copied_layer: Labels, event: Event
    ) -> None:
        """Forward the click event from orthogonal views"""

        if (
            orig_layer is self.source_layer
            and event.type == "mouse_press"
            and "Shift" in event.modifiers
        ):
            # pass the copied layer for extracting the label value, because an orthoview
            # was used
            self.copy_label(event, copied_layer)
