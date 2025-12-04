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

from napari_segmentation_correction.helpers.layer_dropdown import LayerDropdown
from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action,
)


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
        self.source_layer = None
        self.mask_layer = None

        select_delete_box = QGroupBox("Select/delete labels by mask")
        select_delete_box_layout = QVBoxLayout()

        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Labels"))
        self.source_dropdown = LayerDropdown(self.viewer, (Labels))
        self.source_dropdown.layer_changed.connect(self._update_source)
        source_layout.addWidget(self.source_dropdown)

        image2_layout = QHBoxLayout()
        image2_layout.addWidget(QLabel("Mask"))
        self.mask_dropdown = LayerDropdown(self.viewer, (Labels))
        self.mask_dropdown.layer_changed.connect(self._update_mask)
        image2_layout.addWidget(self.mask_dropdown)

        select_delete_box_layout.addLayout(source_layout)
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
        self.select_btn.clicked.connect(lambda: self._select_delete_labels(select=True))
        select_delete_box_layout.addWidget(self.select_btn)

        self.delete_btn = QPushButton("Delete labels")
        self.delete_btn.clicked.connect(
            lambda: self._select_delete_labels(select=False)
        )
        select_delete_box_layout.addWidget(self.delete_btn)

        self.source_dropdown.layer_changed.connect(self._update_buttons)
        self.mask_dropdown.layer_changed.connect(self._update_buttons)
        self._update_buttons()

        select_delete_box.setLayout(select_delete_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(select_delete_box)
        self.setLayout(main_layout)

    def _update_buttons(self) -> None:
        """Update button state according to whether image layers are present"""

        active = (
            self.source_dropdown.selected_layer is not None
            and self.mask_dropdown.selected_layer is not None
        )
        self.select_btn.setEnabled(active)
        self.delete_btn.setEnabled(active)

    def _update_source(self, selected_layer: str) -> None:
        """Update the source layer from which to select/delete labels."""

        if selected_layer == "":
            self.source_layer = None
        else:
            self.source_layer = self.viewer.layers[selected_layer]
            self.source_dropdown.setCurrentText(selected_layer)

        self._update_checkboxes()

    def _update_mask(self, selected_layer: str) -> None:
        """Update the mask layer to indicate which labels should be selected/deleted."""

        if selected_layer == "":
            self.mask_layer = None
        else:
            self.mask_layer = self.viewer.layers[selected_layer]
            self.mask_dropdown.setCurrentText(selected_layer)

        self._update_checkboxes()

    def _update_checkboxes(self) -> None:
        """update the checkbox and buttons as needed needed"""

        if self.mask_layer is not None and self.source_layer is not None:
            self.select_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            if len(self.source_layer.data.shape) == len(self.mask_layer.data.shape) + 1:
                self.stack_checkbox.setEnabled(True)
            else:
                self.stack_checkbox.setEnabled(False)
                self.stack_checkbox.setChecked(False)
            self.edit_in_place.setEnabled(True) if isinstance(
                self.source_layer.data, np.ndarray
            ) else self.edit_in_place.setEnabled(False)
        else:
            self.select_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.stack_checkbox.setEnabled(False)
            self.stack_checkbox.setChecked(False)

    def _select_delete_labels(self, select: bool = True) -> None:
        """Delete labels that overlap with given mask. If the shape of the mask has -1
        dimension compared to the image, the mask will be applied to the current time point
        (index in the first dimension) of the image data."""

        # check data dimensions first
        image_shape = self.source_layer.data.shape
        mask_shape = self.mask_layer.data.shape

        # define action
        action = filter_labels_by_mask if select else delete_labels_by_mask

        # change in place?
        in_place = self.edit_in_place.isChecked() and self.edit_in_place.isEnabled()

        if len(image_shape) == len(mask_shape) + 1 and image_shape[1:] == mask_shape:
            # apply mask to single time point or to full stack depending on checkbox state
            if self.stack_checkbox.isChecked():
                # loop over all time points
                indices = range(self.source_layer.data.shape[0])
                arr = process_action(
                    img1=self.source_layer.data,
                    img2=self.mask_layer.data,
                    img1_index=indices,
                    img2_index=None,
                    action=action,
                    basename=self.source_layer.name,
                    in_place=in_place,
                )

            else:
                tp = self.viewer.dims.current_step[0]
                arr = process_action(
                    img1=self.source_layer.data,
                    img2=self.mask_layer.data,
                    img1_index=tp,
                    img2_index=None,
                    action=action,
                    basename=self.source_layer.name,
                    in_place=in_place,
                )

        elif image_shape == mask_shape:
            indices = None
            if "dimensions" in self.source_layer.metadata:
                dims = self.source_layer.metadata["dimensions"]
                # in the case the use explicitely set the time dimension, we use it to
                # index
                if "T" in dims:
                    indices = range(self.source_layer.data.shape[0])

            arr = process_action(
                img1=self.source_layer.data,
                img2=self.mask_layer.data,
                img1_index=indices,
                img2_index=indices,
                action=action,
                basename=self.source_layer.name,
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
            self.source_layer = self.viewer.add_labels(
                arr,
                name=self.source_layer.name + "_filtered_labels",
                scale=self.source_layer.scale,
            )
        else:
            self.source_layer.refresh()
