import napari
import numpy as np
from napari.layers import Labels
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .layer_dropdown import LayerDropdown
from .process_actions_helpers import process_action


def filter_labels_by_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep labels that touch the mask"""
    to_keep = np.unique(image[mask > 0])
    to_keep_mask = np.isin(image, to_keep)
    image[~to_keep_mask] = 0

    return image


def delete_labels_by_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Delete labels that touch the mask."""
    to_delete = np.unique(image[mask > 0])
    delete_mask = np.isin(image, to_delete)
    image[delete_mask] = 0
    return image


class SelectDeleteMask(QWidget):
    """Widget to select labels to keep or to delete based on overlap with a mask."""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer
        self.image1_layer = None
        self.mask_layer = None
        self.outputdir = None

        ### Add one image to another
        select_delete_box = QGroupBox("Select / Delete labels by mask")
        select_delete_box_layout = QVBoxLayout()

        image1_layout = QHBoxLayout()
        image1_layout.addWidget(QLabel("Labels"))
        self.image1_dropdown = LayerDropdown(self.viewer, (Labels))
        self.image1_dropdown.layer_changed.connect(self._update_image1)
        image1_layout.addWidget(self.image1_dropdown)

        image2_layout = QHBoxLayout()
        image2_layout.addWidget(QLabel("Mask"))
        self.mask_dropdown = LayerDropdown(self.viewer, (Labels))
        self.mask_dropdown.layer_changed.connect(self._update_image2)
        image2_layout.addWidget(self.mask_dropdown)

        select_delete_box_layout.addLayout(image1_layout)
        select_delete_box_layout.addLayout(image2_layout)

        self.stack_checkbox = QCheckBox("Apply 3D mask to all time points in 4D array")
        self.stack_checkbox.setEnabled(False)
        self.edit_in_place = QCheckBox("Edit in place (no undo)")
        self.edit_in_place.setEnabled(False)
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.stack_checkbox)
        options_layout.addWidget(self.edit_in_place)
        select_delete_box_layout.addLayout(options_layout)

        self.select_btn = QPushButton("Select labels")
        self.select_btn.clicked.connect(lambda: self.select_delete_labels(select=True))
        select_delete_box_layout.addWidget(self.select_btn)

        self.delete_btn = QPushButton("Delete labels")
        self.delete_btn.clicked.connect(lambda: self.select_delete_labels(select=False))
        select_delete_box_layout.addWidget(self.delete_btn)

        self.image1_dropdown.layer_changed.connect(self._update_buttons)
        self.mask_dropdown.layer_changed.connect(self._update_buttons)
        self._update_buttons()

        select_delete_box.setLayout(select_delete_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(select_delete_box)
        self.setLayout(main_layout)

    def _update_buttons(self) -> None:
        """Update button state according to whether image layers are present"""

        active = (
            self.image1_dropdown.selected_layer is not None
            and self.mask_dropdown.selected_layer is not None
        )
        self.select_btn.setEnabled(active)
        self.delete_btn.setEnabled(active)

    def _update_image1(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

        # update the checkbox and buttons as needed needed
        if self.mask_layer is not None and self.image1_layer is not None:
            self.select_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            if len(self.image1_layer.data.shape) == len(self.mask_layer.data.shape) + 1:
                self.stack_checkbox.setEnabled(True)
            else:
                self.stack_checkbox.setEnabled(False)
                self.stack_checkbox.setCheckState(False)
            self.edit_in_place.setEnabled(True) if isinstance(
                self.image1_layer.data, np.ndarray
            ) else self.edit_in_place.setEnabled(False)
        else:
            self.select_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.stack_checkbox.setEnabled(False)
            self.stack_checkbox.setCheckState(False)

    def _update_image2(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.mask_layer = None
        else:
            self.mask_layer = self.viewer.layers[selected_layer]
            self.mask_dropdown.setCurrentText(selected_layer)

        # update the checkbox and buttons as needed
        if self.mask_layer is not None and self.image1_layer is not None:
            self.select_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            if len(self.image1_layer.data.shape) == len(self.mask_layer.data.shape) + 1:
                self.stack_checkbox.setEnabled(True)
            else:
                self.stack_checkbox.setEnabled(False)
                self.stack_checkbox.setCheckState(False)
        else:
            self.select_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.stack_checkbox.setEnabled(False)
            self.stack_checkbox.setCheckState(False)

    def select_delete_labels(self, select: bool = True):
        """Delete labels that overlap with given mask. If the shape of the mask has 1 dimension less than the image, the mask will be applied to the current time point (index in the first dimension) of the image data."""

        # check data dimensions first
        image_shape = self.image1_layer.data.shape
        mask_shape = self.mask_layer.data.shape

        # define action
        action = filter_labels_by_mask if select else delete_labels_by_mask

        # change in place?
        in_place = self.edit_in_place.isChecked() and self.edit_in_place.isEnabled()

        if len(image_shape) == len(mask_shape) + 1 and image_shape[1:] == mask_shape:
            # apply mask to single time point or to full stack depending on checkbox state
            if self.stack_checkbox.isChecked():
                # loop over all time points
                indices = range(self.image1_layer.data.shape[0])
                arr = process_action(
                    seg=self.image1_layer.data,
                    mask=self.mask_layer.data,
                    seg_index=indices,
                    mask_index=None,
                    action=action,
                    basename=self.image1_layer.name,
                    in_place=in_place,
                )

            else:
                tp = self.viewer.dims.current_step[0]
                arr = process_action(
                    seg=self.image1_layer.data,
                    mask=self.mask_layer.data,
                    seg_index=tp,
                    mask_index=None,
                    action=action,
                    basename=self.image1_layer.name,
                    in_place=in_place,
                )

        elif image_shape == mask_shape:
            indices = range(self.image1_layer.data.shape[0])
            arr = process_action(
                seg=self.image1_layer.data,
                mask=self.mask_layer.data,
                seg_index=indices,
                mask_index=indices,
                action=action,
                basename=self.image1_layer.name,
                in_place=in_place,
            )

        else:
            msg = QMessageBox()
            msg.setWindowTitle("Images do not have compatible shapes")
            msg.setText("Please provide images that have matching dimensions")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        if not in_place:
            self.image1_layer = self.viewer.add_labels(
                arr,
                name=self.image1_layer.name + "_filtered_labels",
                scale=self.image1_layer.scale,
            )
        else:
            self.image1_layer.refresh()
