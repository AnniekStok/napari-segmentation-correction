import os
import shutil

import dask.array as da
import napari
import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage
from skimage.io import imread
from skimage.measure import label

from .layer_manager import LayerManager


def keep_largest_fragment_per_label(img: np.ndarray, labels: list[int]) -> np.ndarray:
    """Keep only the largest connected component per label in `labels`."""
    out = np.zeros_like(img)
    for label_value in labels:
        if label_value == 0:
            continue
        mask = img == label_value
        if not np.any(mask):
            continue

        labeled, n = ndimage.label(mask)
        if n == 0:
            continue
        if n == 1:
            out[mask] = label_value
            continue

        sizes = np.bincount(labeled.ravel())[1:]  # skip background
        largest_cc = 1 + np.argmax(sizes)  # component index (1-based)
        out[labeled == largest_cc] = label_value

    return out


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

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(
                    self, "Select Output Folder"
                )

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_largest_cluster"),
            )
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
            os.mkdir(outputdir)

            for i in range(
                self.label_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                current_stack = self.label_manager.selected_layer.data[
                    i
                ].compute()  # Compute the current stack
                mask = current_stack > 0
                labeled = label(mask)
                props = np.bincount(labeled.flat)
                props[0] = 0  # ignore background
                largest_label = props.argmax()
                largest_cluster = (labeled == largest_label) * current_stack
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_largest_cluster_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(largest_cluster, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name + "_largest_cluster",
                scale=self.label_manager.selected_layer.scale,
            )
        else:
            shape = self.label_manager.selected_layer.data.shape
            if len(shape) > 3:
                largest_cluster = np.zeros_like(self.label_manager.selected_layer.data)
                for i in range(shape[0]):
                    mask = self.label_manager.selected_layer.data[i] > 0
                    labeled = label(mask)
                    props = np.bincount(labeled.flat)
                    props[0] = 0  # ignore background
                    largest_label = props.argmax()
                    largest_cluster[i] = (
                        labeled == largest_label
                    ) * self.label_manager.selected_layer.data[i]

            else:
                mask = self.label_manager.selected_layer.data > 0
                labeled = label(mask)
                props = np.bincount(labeled.flat)
                props[0] = 0  # ignore background
                largest_label = props.argmax()
                largest_cluster = (
                    labeled == largest_label
                ) * self.label_manager.selected_layer.data

            self.label_manager.selected_layer = self.viewer.add_labels(
                largest_cluster,
                name=self.label_manager.selected_layer.name + "_largest_cluster",
                scale=self.label_manager.selected_layer.scale,
            )

    def _keep_largest_fragment(self):
        """Keep only the largest fragment per label"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(
                    self, "Select Output Folder"
                )

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_largest_fragment"),
            )
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
            os.mkdir(outputdir)

            for i in range(
                self.label_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                current_stack = self.label_manager.selected_layer.data[
                    i
                ].compute()  # Compute the current stack

                labels = np.unique(current_stack)
                largest_fragments = keep_largest_fragment_per_label(
                    current_stack, labels
                )

                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_largest_fragments_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    largest_fragments,
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name + "_largest_fragments",
                scale=self.label_manager.selected_layer.scale,
            )

        else:
            shape = self.label_manager.selected_layer.data.shape
            if len(shape) > 3:
                largest_fragments = np.zeros_like(
                    self.label_manager.selected_layer.data
                )
                for i in range(shape[0]):
                    labels = np.unique(self.label_manager.selected_layer.data[i])
                    largest_fragments[i] = keep_largest_fragment_per_label(
                        self.label_manager.selected_layer.data[i], labels
                    )
            else:
                labels = np.unique(self.label_manager.selected_layer.data)
                largest_fragments = keep_largest_fragment_per_label(
                    self.label_manager.selected_layer.data, labels
                )

            self.label_manager.selected_layer = self.viewer.add_labels(
                largest_fragments,
                name=self.label_manager.selected_layer.name + "_largest_fragments",
                scale=self.label_manager.selected_layer.scale,
            )

    def _conn_comp(self):
        """Run connected component analysis to (re) label the labels array"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(
                    self, "Select Output Folder"
                )

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_conncomp"),
            )
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
            os.mkdir(outputdir)

            for i in range(
                self.label_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                current_stack = self.label_manager.selected_layer.data[
                    i
                ].compute()  # Compute the current stack
                relabeled = label(current_stack)
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_conn_comp_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(relabeled, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name + "_conn_comp",
                scale=self.label_manager.selected_layer.scale,
            )
        else:
            shape = self.label_manager.selected_layer.data.shape
            if len(shape) > 3:
                conn_comp = np.zeros_like(self.label_manager.selected_layer.data)
                for i in range(shape[0]):
                    conn_comp[i] = label(self.label_manager.selected_layer.data[i])

            else:
                conn_comp = label(self.label_manager.selected_layer.data)

            self.label_manager.selected_layer = self.viewer.add_labels(
                conn_comp,
                name=self.label_manager.selected_layer.name + "_conn_comp",
                scale=self.label_manager.selected_layer.scale,
            )
