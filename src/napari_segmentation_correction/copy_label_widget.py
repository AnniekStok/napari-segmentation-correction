from typing import Union

import dask.array as da
import napari
import numpy as np
from napari.layers import Labels
from napari.utils.events import Event
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .layer_dropdown import LayerDropdown


class CopyLabelWidget(QWidget):
    """Widget to create a "Labels Options" Layer from which labels can be copied to another layer"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer

        self.source_layer = None
        self._source_callback = None

        copy_labels_box = QGroupBox("Copy-paste labels")
        copy_labels_layout = QVBoxLayout()

        label = QLabel(
            "Use shift + click on the source layer to copy labels to the target layer. Use right mouse click to copy only a single slice of a label to the target layer."
        )
        label.setWordWrap(True)
        font = label.font()
        font.setItalic(True)
        label.setFont(font)
        copy_labels_layout.addWidget(label)

        self.preserve_label_value = QCheckBox("Preserve label value")
        copy_labels_layout.addWidget(self.preserve_label_value)

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

        copy_labels_box.setLayout(copy_labels_layout)

        layout = QVBoxLayout()
        layout.addWidget(copy_labels_box)
        self.setLayout(layout)

    def _update_source(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

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

    def _update_target(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.target_layer = None
        else:
            self.target_layer = self.viewer.layers[selected_layer]
            self.target_dropdown.setCurrentText(selected_layer)

    def _make_copy_label_callback(self, layer: Labels) -> callable:
        def callback(layer, event):
            if event.type == "mouse_press" and (
                event.button == 2 or "Shift" in event.modifiers
            ):
                coords = layer.world_to_data(event.position)
                coords = [int(c) for c in coords]
                selected_label = layer.get_value(coords)
                if selected_label != 0:
                    self.copy_label(event, coords, selected_label)

        return callback

    def copy_label(self, event: Event, coords: list[int], selected_label: int) -> None:
        """Copy a 2D or 3D label from this layer to a target layer"""

        if self.source_layer is None or self.target_layer is None:
            return
        dims_displayed = event.dims_displayed
        ndims_options = len(self.source_layer.data.shape)
        ndims_label = len(self.target_layer.data.shape)
        ndims = len(coords)

        # ensure minimal bit depth of 16 bit, check later on if 32 bit is needed and convert if asked.
        if (
            self.target_layer.data.dtype == np.int8
            or self.target_layer.data.dtype == np.uint8
        ):
            self.target_layer.data = self.target_layer.data.astype(np.uint16)

        if not (
            ndims_options == ndims_label
            or ndims_options == ndims_label + 1
            or ndims_options == ndims_label - 1
        ):
            msg = QMessageBox()
            msg.setWindowTitle("Invalid dimensions!")
            msg.setText(
                f"Invalid dimensions! Got {ndims_options} for the options layer and {ndims_label} for the target layer."
            )
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        if (
            event.type == "mouse_press" and event.button == 2
        ):  # copy a single slice only
            self.copy_slice_label(
                ndims,
                dims_displayed,
                ndims_options,
                ndims_label,
                coords,
                selected_label,
            )

        elif (
            event.type == "mouse_press" and "Shift" in event.modifiers
        ):  # copy a volume label
            self.copy_volume_label(
                ndims, ndims_options, ndims_label, coords, selected_label
            )

    def copy_volume_label(
        self,
        ndims: int,
        ndims_options: int,
        ndims_label: int,
        coords: list[int],
        selected_label: int,
    ) -> None:
        """Copy a volume label from the options layer to the target layer"""

        options_shape = self.source_layer.data.shape
        labels_shape = self.target_layer.data.shape

        # Always compare the last 2 or 3 dims (y, x or z, y, x)
        spatial_dims = min(3, min(len(options_shape), len(labels_shape)))
        options_spatial = options_shape[-spatial_dims:]
        labels_spatial = labels_shape[-spatial_dims:]

        if options_spatial != labels_spatial:
            msg = QMessageBox()
            msg.setWindowTitle("Invalid dimensions!")
            msg.setText(
                f"The spatial dimensions (z, y, x) of the options layer and the target layer do not match.\n"
                f"Label options layer has shape {options_shape}, target layer has shape {labels_shape}.\n"
                f"Compared last {spatial_dims} dims: {options_spatial} vs {labels_spatial}."
            )
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        # Now calculate the remaining coords for the non-spatial dims
        if spatial_dims == 3:
            remaining_coords = coords[:-(3)]
        elif spatial_dims == 2:
            remaining_coords = coords[:-(2)]
        else:
            remaining_coords = []

        slices = [slice(None)] * ndims
        for i, coord in enumerate(remaining_coords):
            slices[i] = coord

        self._copy(slices, ndims_options, ndims_label, coords, selected_label)

    def copy_slice_label(
        self,
        ndims: int,
        dims_displayed: int,
        ndims_options: int,
        ndims_label: int,
        coords: list[int],
        selected_label: int,
    ) -> None:
        """Copy a single slice of a label from the source layer to the target layer"""

        # Determine how many spatial dims to check (2 or 3)
        spatial_dims = min(
            3, min(len(self.source_layer.data.shape), len(self.target_layer.data.shape))
        )
        options_shape = self.source_layer.data.shape[-spatial_dims:]
        target_shape = self.target_layer.data.shape[-spatial_dims:]

        if options_shape != target_shape:
            msg = QMessageBox()
            msg.setWindowTitle("Invalid dimensions!")
            msg.setText(
                f"The spatial dimensions of the options layer and the target layer do not match.\n"
                f"Label options layer has shape {self.source_layer.data.shape}, target layer has shape {self.target_layer.data.shape}.\n"
                f"Last {spatial_dims} dims: {options_shape} vs {target_shape}."
            )
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        # Create a list of `slice(None)` for all dimensions of self.source_layer.data
        slices = [slice(None)] * ndims
        if ndims_options == ndims_label - 1:
            dims_displayed = [dim - 1 for dim in dims_displayed]
        for i in range(ndims):
            if i not in dims_displayed:
                slices[i] = coords[
                    i
                ]  # Replace the slice with a specific coordinate for slider dims
        self._copy(slices, ndims_options, ndims_label, coords, selected_label)

    def _copy(
        self,
        slices: list[Union[int, slice]],
        ndims_options: int,
        ndims_label: int,
        coords: list[int],
        selected_label: int,
    ) -> None:
        """Copy a label from the options layer to the target layer"""

        # Clip coords to the shape of the label manager's data
        coords_clipped, label_slices = self._clip_coords(
            slices, ndims_options, ndims_label, coords
        )

        # Create mask
        if isinstance(self.source_layer.data, da.core.Array):
            mask = self.source_layer.data[tuple(slices)].compute() == selected_label
        else:
            mask = self.source_layer.data[tuple(slices)] == selected_label

        self._set_label(
            label_slices,
            coords_clipped,
            mask,
            target_label=selected_label
            if self.preserve_label_value.isChecked()
            else None,
        )

        # refresh the layer
        self.target_layer.data = self.target_layer.data

    def _clip_coords(
        self,
        slices: list[Union[int, slice]],
        ndims_options: int,
        ndims_label: int,
        coords: list[int],
    ) -> tuple[list[int], list[int]]:
        """Clip the coordinates to the shape of the target data"""

        if ndims_options == ndims_label + 1:
            coords_clipped = coords[1:]
            label_slices = slices[1:]
        elif ndims_options == ndims_label - 1:
            coords_clipped = [self.viewer.dims.current_step[0], *coords]
            label_slices = [self.viewer.dims.current_step[0], *slices]
        else:
            coords_clipped = coords
            label_slices = slices

        return coords_clipped, label_slices

    def _set_label(
        self,
        label_slices: list[int],
        coords_clipped: list[int],
        mask: np.ndarray,
        target_label: int = None,
    ) -> None:
        """Set the new label in the target stack, either preserving the label value or assigning a new one"""

        # Assign a new label value if None is provided
        if target_label is None:
            target_label = np.max(self.target_layer.data) + 1
            if target_label > 65535 and self.target_layer.data.dtype == np.uint16:
                msg = QMessageBox()
                msg.setWindowTitle("Invalid label!")
                msg.setText(
                    f"Label {target_label} exceeds bit depth! Convert to 32 bit?"
                )
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg.exec_()
                if msg.clickedButton() == msg.button(
                    QMessageBox.Ok
                ):  # Check if Ok was clicked
                    self.target_layer.data = self.target_layer.data.astype(np.uint32)
                elif msg.clickedButton() == msg.button(
                    QMessageBox.Cancel
                ):  # Check if Cancel was clicked
                    return False

        # Get the target layer data
        target_stack = self.target_layer.data

        # Select the correct stack for 3D/4D data
        if isinstance(target_stack, da.core.Array):
            target_stack = target_stack[coords_clipped[0]].compute()
            orig_label = target_stack[tuple(coords_clipped[1:])]
            sliced_data = target_stack[tuple(label_slices[1:])]
        else:
            orig_label = self.target_layer.data[tuple(coords_clipped)]
            sliced_data = self.target_layer.data[tuple(label_slices)]

        orig_mask = sliced_data == orig_label
        sliced_data[orig_mask] = 0
        sliced_data[mask] = target_label

        if isinstance(self.target_layer.data, da.core.Array):
            target_stack[tuple(label_slices[1:])] = sliced_data
            self.target_layer.data[coords_clipped[0]] = target_stack
        else:
            self.target_layer.data[tuple(label_slices)] = sliced_data

    def sync_click(
        self, orig_layer: Labels, copied_layer: Labels, event: Event
    ) -> None:
        """Forward the click event from orthogonal views"""

        if orig_layer is self.source_layer and event.type == "mouse_press":
            selected_label = copied_layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )

            if selected_label != 0:
                coords = orig_layer.world_to_data(event.position)
                coords = [int(c) for c in coords]

                # Process the click event
                self.copy_label(event, coords, selected_label)
