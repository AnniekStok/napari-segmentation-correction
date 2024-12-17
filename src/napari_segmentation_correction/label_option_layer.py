import dask.array as da
import napari
import numpy as np
from qtpy.QtWidgets import (
    QMessageBox,
)

from .layer_manager import LayerManager


class LabelOptions(napari.layers.Labels):
    """Extended labels layer that holds the track information and emits and responds to dynamics visualization signals"""

    @property
    def _type_string(self) -> str:
        return "labels"  # to make sure that the layer is treated as labels layer for saving

    def __init__(
        self,
        viewer: napari.Viewer,
        data: np.array,
        name: str,
        label_manager: LayerManager,
    ):
        super().__init__(
            data=data,
            name=name,
            blending="translucent_no_depth",
        )

        self.contour = 1
        self.viewer = viewer
        self.label_manager = label_manager

        self.mouse_drag_callbacks.append(self._click)

    def _click(self, _, event):
        if event.type == "mouse_press" and (event.button == 2 or "Shift" in event.modifiers):
            coords = self.world_to_data(event.position)
            coords = [int(c) for c in coords]
            selected_label = self.get_value(coords)
            self.copy_label(event, coords, selected_label)

    def copy_label(self, event, coords, selected_label):
        """Copy a 2D or 3D label from this layer to a target layer"""

        dims_displayed = event.dims_displayed
        ndims_options = len(self.data.shape)
        ndims_label = len(self.label_manager.selected_layer.data.shape)
        ndims = len(coords)

        if not (ndims_options == ndims_label or ndims_options == ndims_label + 1 or ndims_options == ndims_label - 1):
            msg = QMessageBox()
            msg.setWindowTitle("Invalid dimensions!")
            msg.setText(f"Invalid dimensions! Got {ndims_options} for the options layer and {ndims_label} for the target layer.")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        if event.type == "mouse_press" and event.button == 2: # copy a single slice only

            # Create a list of `slice(None)` for all dimensions of self.data
            slices = [slice(None)] * ndims
            if ndims_options == ndims_label - 1:
                dims_displayed = [dim-1 for dim in dims_displayed]
            for i in range(ndims):
                if i not in dims_displayed:
                    slices[i] = coords[i]  # Replace the slice with a specific coordinate for slider dims

            # Clip coords to the shape of the label manager's data
            if ndims_options == ndims_label + 1:
                coords_clipped = coords[1:]
                label_slices = slices[1:]
            elif ndims_options == ndims_label - 1:
                coords_clipped = [self.viewer.dims.current_step[0], *coords]
                label_slices = [self.viewer.dims.current_step[0], *slices]
            else:
                coords_clipped = coords
                label_slices = slices

            if isinstance(self.data, da.core.Array):
                mask = self.data[tuple(slices)].compute() == selected_label
            else:
                mask = self.data[tuple(slices)] == selected_label

            # in case we are dealing with a dask array instead of a numpy array
            if isinstance(
                self.label_manager.selected_layer.data, da.core.Array
            ):
                target_stack = self.label_manager.selected_layer.data[
                    coords_clipped[0]
                ].compute()

                sliced_data = target_stack[tuple(label_slices[1:])]
                new_selected_label = np.max(target_stack) + 1
                orig_label = target_stack[tuple(coords_clipped[1:])]
                orig_mask = sliced_data == orig_label  # Mask must have the same shape as sliced_data
                sliced_data[orig_mask] = 0
                sliced_data[mask] = new_selected_label
                target_stack[tuple(label_slices[1:])] = sliced_data
                self.label_manager.selected_layer.data[coords_clipped[0]] = target_stack

            else:
                new_selected_label = np.max(self.label_manager.selected_layer.data) + 1
                orig_label = self.label_manager.selected_layer.data[tuple(coords_clipped)]
                sliced_data = self.label_manager.selected_layer.data[tuple(label_slices)]

                # Create the mask for the original label
                orig_mask = sliced_data == orig_label  # Mask must have the same shape as sliced_data

                # Modify only the selected slice with the mask
                sliced_data[orig_mask] = 0
                sliced_data[mask] = new_selected_label

                # Assign the modified slice back to the original data
                self.label_manager.selected_layer.data[tuple(label_slices)] = sliced_data

            self.label_manager.selected_layer.data = (
                self.label_manager.selected_layer.data
            ) # to refresh the layer

        elif event.type == "mouse_press" and "Shift" in event.modifiers:

            options_shape = self.data.shape
            labels_shape = self.label_manager.selected_layer.data.shape

            overlapping_dims = []
            for i in range(min(ndims_options, ndims_label)):
                if options_shape[-i-1] == labels_shape[-i-1]:
                    overlapping_dims.append(options_shape[-i-1])
                else:
                    break
                overlapping_dims.reverse()

            # we copy the overlapping dims, but max 3
            slices = [slice(None)] * ndims
            if len(overlapping_dims) >= 3:
                remaining_coords = coords[:-3]
            elif len(overlapping_dims) == 2:
                remaining_coords = coords[:-2]
            for i, coord in enumerate(remaining_coords):
                slices[i] = coord

            # Clip coords to the shape of the label manager's data
            if ndims_options == ndims_label + 1:
                coords_clipped = coords[1:]
                label_slices = slices[1:]
            elif ndims_options == ndims_label - 1:
                coords_clipped = [self.viewer.dims.current_step[0], *coords]
                label_slices = [self.viewer.dims.current_step[0], *slices]
            else:
                coords_clipped = coords
                label_slices = slices

            if isinstance(self.data, da.core.Array):
                mask = self.data[tuple(slices)].compute() == selected_label
            else:
                mask = self.data[tuple(slices)] == selected_label

            if isinstance(
                self.label_manager.selected_layer.data, da.core.Array
            ):
                target_stack = self.label_manager.selected_layer.data[
                    coords_clipped[0]
                ].compute()

                sliced_data = target_stack[tuple(label_slices[1:])]
                new_selected_label = np.max(target_stack) + 1
                orig_label = target_stack[tuple(coords_clipped[1:])]
                orig_mask = sliced_data == orig_label  # Mask must have the same shape as sliced_data
                sliced_data[orig_mask] = 0
                sliced_data[mask] = new_selected_label
                target_stack[tuple(label_slices[1:])] = sliced_data
                self.label_manager.selected_layer.data[coords_clipped[0]] = target_stack

            else:
                new_selected_label = np.max(self.label_manager.selected_layer.data) + 1
                orig_label = self.label_manager.selected_layer.data[tuple(coords_clipped)]
                sliced_data = self.label_manager.selected_layer.data[tuple(label_slices)]

                # Create the mask for the original label
                orig_mask = sliced_data == orig_label  # Mask must have the same shape as sliced_data

                # Modify only the selected slice with the mask
                sliced_data[orig_mask] = 0
                sliced_data[mask] = new_selected_label

                # Assign the modified slice back to the original data
                self.label_manager.selected_layer.data[tuple(label_slices)] = sliced_data
            self.label_manager.selected_layer.data = self.label_manager.selected_layer.data
