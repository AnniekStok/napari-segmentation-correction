import copy

import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import label, regionprops_table

from .layer_manager import LayerManager
from .process_actions_helpers import process_action_seg


def keep_largest_cluster(img: np.ndarray) -> np.ndarray:
    """Keep the largest connected cluster of labels"""

    mask = img > 0
    labeled = label(mask)
    props = np.bincount(labeled.flat)
    props[0] = 0  # ignore background
    largest_label = props.argmax()
    return (labeled == largest_label) * img


def keep_largest_fragment_per_label(img: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component per label."""

    relabeled = label(img)
    df = pd.DataFrame(
        regionprops_table(
            relabeled,
            intensity_image=img,
            properties=("label", "intensity_mean", "num_pixels"),
        )
    )
    df_reduced = df.loc[df.groupby("intensity_mean")["num_pixels"].idxmax()]
    remaining_labels = list(df_reduced["label"])
    to_keep_mask = np.isin(relabeled, remaining_labels)
    out = copy.deepcopy(img)
    out[~to_keep_mask] = 0

    return out


def connected_component_labeling(img: np.ndarray) -> np.ndarray:
    """Run connected components labeling"""

    return label(img)


class ConnectedComponents(QWidget):
    """Widget to run connected component analysis"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        conn_comp_box = QGroupBox("Connected Component Analysis")
        conn_comp_box_layout = QVBoxLayout()

        self.conncomp_btn = QPushButton("Find connected components")
        self.conncomp_btn.setToolTip(
            "Run connected component analysis to (re)label the labels layer"
        )
        self.conncomp_btn.clicked.connect(self._conn_comp)
        self.conncomp_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        self.label_manager.layer_update.connect(self._update_button_state)

        conn_comp_box_layout.addWidget(self.conncomp_btn)

        self.keep_largest_btn = QPushButton("Keep largest component cluster")
        self.keep_largest_btn.setToolTip(
            "Keep only the labels part of the largest non-zero connected component"
        )
        self.keep_largest_btn.clicked.connect(self._keep_largest_cluster)
        self.keep_largest_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        conn_comp_box_layout.addWidget(self.keep_largest_btn)

        self.keep_largest_fragment_btn = QPushButton("Keep largest fragment per label")
        self.keep_largest_fragment_btn.setToolTip(
            "For each label, keep only the largest connected fragment"
        )
        self.keep_largest_fragment_btn.clicked.connect(self._keep_largest_fragment)
        self.keep_largest_fragment_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        conn_comp_box_layout.addWidget(self.keep_largest_fragment_btn)

        conn_comp_box.setLayout(conn_comp_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(conn_comp_box)
        self.setLayout(main_layout)

    def _update_button_state(self):
        self.conncomp_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        self.keep_largest_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )
        self.keep_largest_fragment_btn.setEnabled(
            isinstance(self.label_manager.selected_layer, napari.layers.Labels)
        )

    def _keep_largest_cluster(self):
        """Keep only the labels part of the largest non-zero connected component"""

        action = keep_largest_cluster
        largest_cluster = process_action_seg(
            self.label_manager.selected_layer.data,
            action,
            basename=self.label_manager.selected_layer.name,
        )
        self.label_manager.selected_layer = self.viewer.add_labels(
            largest_cluster,
            name=self.label_manager.selected_layer.name + "_largest_cluster",
            scale=self.label_manager.selected_layer.scale,
        )

    def _keep_largest_fragment(self):
        """Keep only the largest fragment per label"""

        action = keep_largest_fragment_per_label
        largest_frags = process_action_seg(
            self.label_manager.selected_layer.data,
            action,
            basename=self.label_manager.selected_layer.name,
        )
        self.label_manager.selected_layer = self.viewer.add_labels(
            largest_frags,
            name=self.label_manager.selected_layer.name + "_largest_fragment_per_label",
            scale=self.label_manager.selected_layer.scale,
        )

    def _conn_comp(self):
        """Run connected component analysis to (re) label the labels array"""

        action = connected_component_labeling
        conncomp = process_action_seg(
            self.label_manager.selected_layer.data,
            action,
            basename=self.label_manager.selected_layer.name,
        )
        self.label_manager.selected_layer = self.viewer.add_labels(
            conncomp,
            name=self.label_manager.selected_layer.name + "_conncomp",
            scale=self.label_manager.selected_layer.scale,
        )
