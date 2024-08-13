
import functools

import dask.array as da
import napari
import numpy as np
from napari.layers import Labels, Points
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._layer_dropdown import LayerDropdown
from .layer_manager import LayerManager

class PointFilter(QWidget):
    """Use a points layer to remove or keep selected labels"""

    def __init__(self, viewer: "napari.viewer.Viewer", label_manager: LayerManager) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager

        point_filter_box = QGroupBox("Select objects with points")
        point_filter_layout = QVBoxLayout()
        self.point_dropdown = LayerDropdown(self.viewer, (Points))
        self.point_dropdown.layer_changed.connect(self._update_points)

        remove_keep_btn_layout = QHBoxLayout()
        self.keep_pts_btn = QPushButton("Keep")
        self.keep_pts_btn.clicked.connect(self._keep_objects)
        self.remove_pts_btn = QPushButton("Remove")
        self.remove_pts_btn.clicked.connect(self._delete_objects)
        remove_keep_btn_layout.addWidget(self.keep_pts_btn)
        remove_keep_btn_layout.addWidget(self.remove_pts_btn)

        point_filter_layout.addWidget(self.point_dropdown)
        point_filter_layout.addLayout(remove_keep_btn_layout)

        point_filter_box.setLayout(point_filter_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(point_filter_box)
        self.setLayout(main_layout)

    def _keep_objects(self) -> None:
        """Keep only the labels that are selected by the points layer."""

        if self.label_manager.selected_layer is not None:
            if isinstance(self.label_manager.selected_layer.data, da.core.Array):
                tps = np.unique([int(p[0]) for p in self.points.data])
                for tp in tps:
                    labels_to_keep = []
                    points = [p for p in self.points.data if p[0] == tp]
                    current_stack = self.label_manager.selected_layer.data[
                        tp
                    ].compute()  # Compute the current stack
                    for p in points:
                        labels_to_keep.append(
                            current_stack[int(p[1]), int(p[2]), int(p[3])]
                        )
                    mask = functools.reduce(
                        np.logical_or,
                        (current_stack == val for val in labels_to_keep),
                    )
                    filtered = np.where(mask, current_stack, 0)
                    self.label_manager.selected_layer.data[tp] = filtered
                self.label_manager.selected_layer.data = self.label_manager.selected_layer.data  # to trigger viewer update

            else:
                if len(self.points.data[0]) == 4:
                    tps = np.unique([int(p[0]) for p in self.points.data])
                    for tp in tps:
                        labels_to_keep = []
                        points = [p for p in self.points.data if p[0] == tp]
                        for p in points:
                            labels_to_keep.append(
                                self.label_manager.selected_layer.data[
                                    tp, int(p[1]), int(p[2]), int(p[3])
                                ]
                            )
                        mask = functools.reduce(
                            np.logical_or,
                            (
                                self.label_manager.selected_layer.data[tp] == val
                                for val in labels_to_keep
                            ),
                        )
                        filtered = np.where(mask, self.label_manager.selected_layer.data[tp], 0)
                        self.label_manager.selected_layer.data[tp] = filtered
                    self.label_manager.selected_layer.data = self.label_manager.selected_layer.data  # to trigger viewer update

                else:
                    labels_to_keep = []
                    for p in self.points.data:
                        if len(p) == 2:
                            labels_to_keep.append(
                                self.label_manager.selected_layer.data[int(p[0]), int(p[1])]
                            )
                        elif len(p) == 3:
                            labels_to_keep.append(
                                self.label_manager.selected_layer.data[int(p[0]), int(p[1]), int(p[2])]
                            )

                    mask = functools.reduce(
                        np.logical_or,
                        (self.label_manager.selected_layer.data == val for val in labels_to_keep),
                    )
                    filtered = np.where(mask, self.label_manager.selected_layer.data, 0)

                    self.label_manager.selected_layer = self.viewer.add_labels(
                        filtered, name=self.label_manager.selected_layer.name + "_points_kept"
                    )
                    self._update_labels(self.label_manager.selected_layer.name)

    def _update_points(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'points' layer for picking labels."""

        if selected_layer == "":
            self.points = None
        else:
            self.points = self.viewer.layers[selected_layer]
            self.point_dropdown.setCurrentText(selected_layer)

    def _delete_objects(self) -> None:
        """Delete all labels selected by the points layer."""

        if self.label_manager.selected_layer is not None:
            if isinstance(self.label_manager.selected_layer.data, da.core.Array):
                tps = np.unique([int(p[0]) for p in self.points.data])
                for tp in tps:
                    labels_to_keep = []
                    points = [p for p in self.points.data if p[0] == tp]
                    current_stack = self.label_manager.selected_layer.data[
                        tp
                    ].compute()  # Compute the current stack
                    for p in points:
                        labels_to_keep.append(
                            current_stack[int(p[1]), int(p[2]), int(p[3])]
                        )
                    mask = functools.reduce(
                        np.logical_or,
                        (current_stack == val for val in labels_to_keep),
                    )
                    inverse_mask = np.logical_not(mask)
                    filtered = np.where(inverse_mask, current_stack, 0)
                    self.label_manager.selected_layer.data[tp] = filtered
                self.label_manager.selected_layer.data = self.label_manager.selected_layer.data

            else:
                if len(self.points.data[0]) == 4:
                    tps = np.unique([int(p[0]) for p in self.points.data])
                    for tp in tps:
                        labels_to_keep = []
                        points = [p for p in self.points.data if p[0] == tp]
                        for p in points:
                            labels_to_keep.append(
                                self.label_manager.selected_layer.data[
                                    tp, int(p[1]), int(p[2]), int(p[3])
                                ]
                            )
                        mask = functools.reduce(
                            np.logical_or,
                            (
                                self.label_manager.selected_layer.data[tp] == val
                                for val in labels_to_keep
                            ),
                        )
                        inverse_mask = np.logical_not(mask)
                        filtered = np.where(inverse_mask, self.label_manager.selected_layer.data[tp], 0)
                        self.label_manager.selected_layer.data[tp] = filtered
                    self.label_manager.selected_layer.data = self.label_manager.selected_layer.data  # to trigger viewer update

                else:
                    labels_to_keep = []
                    for p in self.points.data:
                        if len(p) == 2:
                            labels_to_keep.append(
                                self.label_manager.selected_layer.data[int(p[0]), int(p[1])]
                            )
                        elif len(p) == 3:
                            labels_to_keep.append(
                                self.label_manager.selected_layer.data[int(p[0]), int(p[1]), int(p[2])]
                            )
                        elif len(p) == 4:
                            labels_to_keep.append(
                                self.label_manager.selected_layer.data[
                                    int(p[0]), int(p[1]), int(p[2], int(p[3]))
                                ]
                            )

                    mask = functools.reduce(
                        np.logical_or,
                        (self.label_manager.selected_layer.data == val for val in labels_to_keep),
                    )
                    inverse_mask = np.logical_not(mask)
                    filtered = np.where(inverse_mask, self.label_manager.selected_layer.data, 0)

                    self.label_manager.selected_layer = self.viewer.add_labels(
                        filtered, name=self.label_manager.selected_layer.name + "_points_removed"
                    )
                    self._update_labels(self.label_manager.selected_layer.name)
