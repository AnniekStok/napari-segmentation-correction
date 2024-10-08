import dask.array as da
import napari
import numpy as np
from qtpy.QtWidgets import (
    QMessageBox,
)

from .layer_manager import LayerManager


class LabelOptions(napari.layers.Labels):
    """Extended labels layer that holds the track information and emits and responds to dynamics visualization signals"""

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

        self.viewer = viewer
        self.label_manager = label_manager

        @self.mouse_drag_callbacks.append
        def cell_copied(layer, event):
            if event.type == "mouse_press" and event.button == 2:
                coords = self.world_to_data(event.position)
                coords = [int(c) for c in coords]
                selected_label = self.get_value(coords)

                ndims = len(self.data.shape)
                if ndims == 5:
                    mask = (
                        self.data[coords[0], coords[1], coords[2], :, :] == selected_label
                    )
                elif ndims == 4:
                    mask = (
                        self.data[coords[0], coords[1], :, :] == selected_label
                    )
                elif ndims == 3: 
                    mask = (
                        self.data[coords[0], :, :] == selected_label
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
                    orig_label = target_stack[
                        coords[-3], coords[-2], coords[-1]
                    ]
                    if orig_label != 0:
                        target_stack[target_stack == orig_label] = 0
                    target_stack[mask] = np.max(target_stack) + 1
                    self.label_manager.selected_layer.data[coords[-4]] = (
                        target_stack
                    )
                    self.label_manager.selected_layer.data = (
                        self.label_manager.selected_layer.data
                    )

                else:
                    if len(self.label_manager.selected_layer.data.shape) == 3:
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-3], coords[-2], coords[-1]
                        ]
                       
                        if orig_label != 0:
                            self.label_manager.selected_layer.data[coords[-3]][
                                self.label_manager.selected_layer.data[coords[-3]] == orig_label] = 0  # set the original label to zero in current slice only
                        self.label_manager.selected_layer.data[coords[-3]][mask] = (
                            np.max(self.label_manager.selected_layer.data) + 1
                        )
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
                            self.label_manager.selected_layer.data[coords[-4]][coords[-3]][
                                self.label_manager.selected_layer.data[coords[-4]][coords[-3]] == orig_label] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[coords[-4]][coords[-3]][
                            mask
                        ] = (
                            np.max(self.label_manager.selected_layer.data) + 1
                        )
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
            elif event.type == "mouse_press" and "Shift" in event.modifiers:
                coords = self.world_to_data(event.position)
                coords = [int(c) for c in coords]
                selected_label = self.get_value(coords)

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
                    orig_label = target_stack[
                        coords[-3], coords[-2], coords[-1]
                    ]
                    if orig_label != 0:
                        target_stack[target_stack == orig_label] = 0
                    target_stack[mask] = np.max(target_stack) + 1
                    self.label_manager.selected_layer.data[coords[-4]] = (
                        target_stack
                    )
                    self.label_manager.selected_layer.data = (
                        self.label_manager.selected_layer.data
                    )

                else:
                    if len(self.label_manager.selected_layer.data.shape) == 3:
                        orig_label = self.label_manager.selected_layer.data[
                            coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.label_manager.selected_layer.data[
                                self.label_manager.selected_layer.data
                                == orig_label
                            ] = 0  # set the original label to zero
                        self.label_manager.selected_layer.data[mask] = (
                            np.max(self.label_manager.selected_layer.data) + 1
                        )
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
                        ] = (
                            np.max(self.label_manager.selected_layer.data) + 1
                        )
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
