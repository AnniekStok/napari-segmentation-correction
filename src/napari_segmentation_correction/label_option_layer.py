import dask.array as da
import napari
import numpy as np
from qtpy.QtWidgets import (
    QMessageBox,
)

from .layer_manager import LayerManager
from typing import Union
from napari.utils.events import Event
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
        """Click event handler to copy slice or volume labels from the options layer to the target layer"""
        if event.type == "mouse_press" and (event.button == 2 or "Shift" in event.modifiers):
            coords = self.world_to_data(event.position)
            coords = [int(c) for c in coords]
            selected_label = self.get_value(coords)
            if selected_label != 0: # do not copy background label
                self.copy_label(event, coords, selected_label)

    def copy_label(self, event: Event, coords: list[int], selected_label: int) -> None:
        """Copy a 2D or 3D label from this layer to a target layer"""
       
        dims_displayed = event.dims_displayed
        ndims_options = len(self.data.shape)
        ndims_label = len(self.label_manager.selected_layer.data.shape)
        ndims = len(coords)

        # ensure minimal bit depth of 16 bit, check later on if 32 bit is needed and convert if asked.
        if self.label_manager.selected_layer.data.dtype == np.int8 or self.label_manager.selected_layer.data.dtype == np.uint8:
            self.label_manager.selected_layer.data = self.label_manager.selected_layer.data.astype(np.uint16)

        if not (ndims_options == ndims_label or ndims_options == ndims_label + 1 or ndims_options == ndims_label - 1):
            msg = QMessageBox()
            msg.setWindowTitle("Invalid dimensions!")
            msg.setText(f"Invalid dimensions! Got {ndims_options} for the options layer and {ndims_label} for the target layer.")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        if event.type == "mouse_press" and event.button == 2: # copy a single slice only
            self.copy_slice_label(ndims, dims_displayed, ndims_options, ndims_label, coords, selected_label)

        elif event.type == "mouse_press" and "Shift" in event.modifiers: # copy a volume label
            self.copy_volume_label(ndims, ndims_options, ndims_label, coords, selected_label)
        
        # refresh the layer
        self.label_manager.selected_layer.data = (
            self.label_manager.selected_layer.data
        )

    def copy_volume_label(self, ndims: int, ndims_options: int, ndims_label: int, coords: list[int], selected_label: int) -> None:
        """Copy a volume label from the options layer to the target layer"""

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

        self.copy(slices, ndims_options, ndims_label, coords, selected_label)
   
    def copy_slice_label(self, ndims: int, dims_displayed: int, ndims_options: int, ndims_label: int, coords: list[int], selected_label: int) -> None:

        # Create a list of `slice(None)` for all dimensions of self.data
            slices = [slice(None)] * ndims
            if ndims_options == ndims_label - 1:
                dims_displayed = [dim-1 for dim in dims_displayed]
            for i in range(ndims):
                if i not in dims_displayed:
                    slices[i] = coords[i]  # Replace the slice with a specific coordinate for slider dims
        
            self.copy(slices, ndims_options, ndims_label, coords, selected_label)

    def copy(self, slices: list[Union[int, slice]], ndims_options: int, ndims_label: int, coords: list[int], selected_label: int) -> None:
        """Copy a label from the options layer to the target layer"""
        
        # Clip coords to the shape of the label manager's data
        coords_clipped, label_slices = self.clip_coords(slices, ndims_options, ndims_label, coords)

        # Create mask
        if isinstance(self.data, da.core.Array):
            mask = self.data[tuple(slices)].compute() == selected_label
        else:
            mask = self.data[tuple(slices)] == selected_label

        # check whether we are dealing with a dask array or a numpy array and process accordingly
        if isinstance(
            self.label_manager.selected_layer.data, da.core.Array
        ):
            self.set_label_dask_array(label_slices, coords_clipped, mask)
        else:
            self.set_label_numpy_array(label_slices, coords_clipped, mask)

    def clip_coords(self, slices: list[Union[int, slice]], ndims_options: int, ndims_label: int, coords: list[int]) -> tuple[list[int], list[int]]: 
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
    
    def set_label_numpy_array(self, label_slices: list[int], coords_clipped: list[int], mask: np.ndarray) -> None:
        """Set the new label in the target stack of a numpy array"""

        new_selected_label = np.max(self.label_manager.selected_layer.data) + 1
        if new_selected_label > 65535 and self.label_manager.selected_layer.data.dtype == np.uint16:
            msg = QMessageBox()
            msg.setWindowTitle("Invalid label!")
            msg.setText(f"Label {new_selected_label} exceeds bit depth! Convert to 32 bit?")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.exec_()
            if msg.clickedButton() == msg.button(QMessageBox.Ok):  # Check if Ok was clicked
                self.label_manager.selected_layer.data = self.label_manager.selected_layer.data.astype(np.uint32)
            elif msg.clickedButton() == msg.button(QMessageBox.Cancel):  # Check if Cancel was clicked
                return False
        
        orig_label = self.label_manager.selected_layer.data[tuple(coords_clipped)]
        sliced_data = self.label_manager.selected_layer.data[tuple(label_slices)]

        # Create the mask for the original label
        orig_mask = sliced_data == orig_label  # Mask must have the same shape as sliced_data

        # Modify only the selected slice with the mask
        sliced_data[orig_mask] = 0
        sliced_data[mask] = new_selected_label

        # Assign the modified slice back to the original data
        self.label_manager.selected_layer.data[tuple(label_slices)] = sliced_data
    
    def set_label_dask_array(self, label_slices: list[int], coords_clipped: list[int], mask: np.ndarray) -> None:
        """Set the label in the target stack for dask arrays"""

        target_stack = self.label_manager.selected_layer.data[
            coords_clipped[0]
        ].compute()
        sliced_data = target_stack[tuple(label_slices[1:])]
        new_selected_label = np.max(target_stack) + 1
        if new_selected_label > 65535 and target_stack.dtype == np.uint16:
            msg = QMessageBox()
            msg.setWindowTitle("Invalid label!")
            msg.setText(f"Label {new_selected_label} exceeds bit depth! Convert to 32 bit?")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.exec_()
            if msg.clickedButton() == msg.button(QMessageBox.Ok):  # Check if Ok was clicked
                target_stack = target_stack.astype(np.uint32)
                self.label_manager.selected_layer.data = self.label_manager.selected_layer.data.astype(np.uint32)  # Convert entire layer to 32 bit
            elif msg.clickedButton() == msg.button(QMessageBox.Cancel):  # Check if Cancel was clicked
                return False
            
        orig_label = target_stack[tuple(coords_clipped[1:])]
        orig_mask = sliced_data == orig_label  # Mask must have the same shape as sliced_data
        sliced_data[orig_mask] = 0
        sliced_data[mask] = new_selected_label
        target_stack[tuple(label_slices[1:])] = sliced_data
        self.label_manager.selected_layer.data[coords_clipped[0]] = target_stack