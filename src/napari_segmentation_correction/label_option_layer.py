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

            dims_displayed = event.dims_displayed
            if event.type == "mouse_press" and event.button == 2: # copy a single slice only

                ndims = len(self.data.shape)
                ndims_label = len(self.label_manager.selected_layer.data.shape)

                if not (ndims == ndims_label or ndims == ndims_label + 1):
                    print(f"Invalid dimensions! Got {ndims} for the options layer and {ndims_label} for the target layer.")
                    return

                # Create a list of `slice(None)` for all dimensions of self.data
                slices = [slice(None)] * ndims  
                for i in range(ndims):
                    if i not in dims_displayed:
                        slices[i] = coords[i]  # Replace the slice with a specific coordinate for slider dims

                # Clip coords to the shape of the label manager's data
                if ndims == ndims_label + 1:
                    coords_clipped = coords[1:]
                    label_slices = slices[1:]
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
                ndims = len(self.data.shape)
                if ndims == 5:
                    mask = (
                        self.data[coords[0], coords[1], :, :, :] == selected_label
                    )
                elif ndims == 4:
                    mask = (
                        self.data[coords[0], :, :, :] == selected_label
                    )
                elif ndims == 3:
                    mask = (
                        self.data[:, :, :] == selected_label
                    )
                else:
                    print('This number of dimensions is currently not supported', ndims)
                    return

                if isinstance(
                    self.label_manager.selected_layer.data, da.core.Array
                ):
                    target_stack = self.label_manager.selected_layer.data[
                        coords[-4]
                    ].compute()

                    new_selected_label = np.max(target_stack) + 1
                    orig_label = target_stack[
                        coords[-3], coords[-2], coords[-1]
                    ]
                    if orig_label != 0:
                        target_stack[target_stack == orig_label] = 0
                    target_stack[mask] = new_selected_label
                    self.label_manager.selected_layer.data[coords[-4]] = (
                        target_stack
                    )
                    self.label_manager.selected_layer.data = (
                        self.label_manager.selected_layer.data
                    )

                else:
                    new_selected_label = np.max(self.label_manager.selected_layer.data) + 1
                    if len(self.label_manager.selected_layer.data.shape) == 3:
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.label_manager.selected_layer.data[
                                self.label_manager.selected_layer.data
                                == orig_label
                            ] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[mask] = new_selected_label
                        self.label_manager.selected_layer.data = (
                            self.label_manager.selected_layer.data
                        )

                    elif (
                        len(self.label_manager.selected_layer.data.shape) == 4
                    ):
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-4], coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.label_manager.selected_layer.data[coords[-4]][
                                self.label_manager.selected_layer.data[
                                    coords[-4]
                                ]
                                == orig_label
                            ] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[coords[-4]][
                            mask
                        ] = new_selected_label
                        self.label_manager.selected_layer.data = (
                            self.label_manager.selected_layer.data
                        )

                    elif (
                        len(self.label_manager.selected_layer.data.shape) == 5
                    ):
                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setText(
                            "Copy-pasting in 5 dimensions is not implemented, do you want to convert the labels layer to 5 dimensions (tzyx)?"
                        )
                        msg_box.setWindowTitle("Convert to 4 dimensions?")

                        yes_button = msg_box.addButton(QMessageBox.Yes)
                        no_button = msg_box.addButton(QMessageBox.No)

                        msg_box.exec_()

                        if msg_box.clickedButton() == yes_button:
                            self.label_manager.selected_layer.data = (
                                self.label_manager.selected_layer.data[0]
                            )
                        elif msg_box.clickedButton() == no_button:
                            return False
                    else:
                        print(
                            "copy-pasting in more than 5 dimensions is not supported"
                        )
