import dask.array as da
import napari
import numpy as np
import copy
import functools
from napari.layers import Labels
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .layer_dropdown import LayerDropdown


class SelectDeleteMask(QWidget):
    """Widget to perform calculations between two images"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer

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
        self.image2_dropdown = LayerDropdown(self.viewer, (Labels))
        self.image2_dropdown.layer_changed.connect(self._update_image2)
        image2_layout.addWidget(self.image2_dropdown)

        select_delete_box_layout.addLayout(image1_layout)
        select_delete_box_layout.addLayout(image2_layout)

        select_btn = QPushButton("Select labels")
        select_btn.clicked.connect(self.select_labels)
        select_delete_box_layout.addWidget(select_btn)

        delete_btn = QPushButton("Delete labels")
        delete_btn.clicked.connect(self.delete_labels)
        select_delete_box_layout.addWidget(delete_btn)

        select_delete_box.setLayout(select_delete_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(select_delete_box)
        self.setLayout(main_layout)

    def _update_image1(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

    def _update_image2(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image2_layer = None
        else:
            self.image2_layer = self.viewer.layers[selected_layer]
            self.image2_dropdown.setCurrentText(selected_layer)

    def select_labels(self):
        """Select labels that overlap with the given mask"""

        if isinstance(self.image1_layer, da.core.Array) or isinstance(
            self.image2_layer, da.core.Array
        ):
            msg = QMessageBox()
            msg.setWindowTitle(
                "Cannot run this operation on dask arrays"
            )
            msg.setText("Cannot run this operation on dask arrays")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        if self.image1_layer.data.shape != self.image2_layer.data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Images must have the same shape")
            msg.setText("Images must have the same shape")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        
        to_keep = np.unique(self.image1_layer.data[self.image2_layer.data > 0])
        filtered_mask = functools.reduce(np.logical_or, (self.image1_layer.data==val for val in to_keep))
        self.viewer.add_labels(np.where(filtered_mask, self.image1_layer.data, 0), name="selected labels")

    def delete_labels(self):
        """Delete labels that overlap with given mask """

        if isinstance(self.image1_layer, da.core.Array) or isinstance(
            self.image2_layer, da.core.Array
        ):
            msg = QMessageBox()
            msg.setWindowTitle(
                "Cannot run this operation on dask arrays"
            )
            msg.setText("Cannot run this operation on dask arrays")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        if self.image1_layer.data.shape != self.image2_layer.data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Images must have the same shape")
            msg.setText("Images must have the same shape")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        
        to_delete = np.unique(self.image1_layer.data[self.image2_layer.data > 0])
        selected_labels = self.viewer.add_labels(copy.deepcopy(self.image1_layer.data), name="selected_self.image1_layer.data")
        for label in to_delete:
            selected_labels.data[selected_labels.data == label] = 0
