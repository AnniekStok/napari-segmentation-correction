import copy

import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
)
from skimage.measure import label, regionprops_table

from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action_seg,
)
from napari_segmentation_correction.tool_widgets.base_tool_widget import BaseToolWidget


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


class ConnectedComponents(BaseToolWidget):
    """Widget for various connected component analysis functions."""

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(napari.layers.Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        box = QGroupBox("Connected Component Analysis")
        box_layout = QVBoxLayout()

        # connected components labeling
        self.conncomp_btn = QPushButton("Find connected components")
        self.conncomp_btn.setToolTip(
            "Run connected component analysis to (re)label the labels layer"
        )
        self.conncomp_btn.clicked.connect(self._conn_comp)
        box_layout.addWidget(self.conncomp_btn)

        # keep the largest connected cluster of labels
        self.keep_largest_btn = QPushButton("Keep largest component cluster")
        self.keep_largest_btn.setToolTip(
            "Keep only the labels part of the largest non-zero connected component"
        )
        self.keep_largest_btn.clicked.connect(self._keep_largest_cluster)
        box_layout.addWidget(self.keep_largest_btn)

        # keep the largest fragment per label
        self.keep_largest_fragment_btn = QPushButton("Keep largest fragment per label")
        self.keep_largest_fragment_btn.setToolTip(
            "For each label, keep only the largest connected fragment"
        )
        self.keep_largest_fragment_btn.clicked.connect(self._keep_largest_fragment)
        box_layout.addWidget(self.keep_largest_fragment_btn)

        # update button state
        self._update_button_state()
        self.update_status.connect(self._update_button_state)

        box.setLayout(box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _update_button_state(self) -> None:
        """Activate/deactivate the buttons according to whether the current layer is of
        the correct type."""

        state = self.layer is not None
        self.conncomp_btn.setEnabled(state)
        self.keep_largest_btn.setEnabled(state)
        self.keep_largest_fragment_btn.setEnabled(state)

    def _keep_largest_cluster(self) -> None:
        """Keep only the labels part of the largest non-zero connected component"""

        action = keep_largest_cluster
        largest_cluster = process_action_seg(
            self.layer.data,
            action,
            basename=self.layer.name,
        )

        if largest_cluster is not None:
            self.layer = self.viewer.add_labels(
                largest_cluster,
                name=self.layer.name + "_largest_cluster",
                scale=self.layer.scale,
            )

    def _keep_largest_fragment(self) -> None:
        """Keep only the largest fragment per label"""

        action = keep_largest_fragment_per_label
        largest_frags = process_action_seg(
            self.layer.data,
            action,
            basename=self.layer.name,
        )

        if largest_frags is not None:
            self.layer = self.viewer.add_labels(
                largest_frags,
                name=self.layer.name + "_largest_fragment",
                scale=self.layer.scale,
            )

    def _conn_comp(self) -> None:
        """Run connected component analysis to (re)label the labels array"""

        action = connected_component_labeling
        conncomp = process_action_seg(
            self.layer.data,
            action,
            basename=self.layer.name,
        )

        if conncomp is not None:
            self.layer = self.viewer.add_labels(
                conncomp,
                name=self.layer.name + "_conncomp",
                scale=self.layer.scale,
            )
