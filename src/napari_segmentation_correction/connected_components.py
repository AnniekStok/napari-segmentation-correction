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
from skimage.io import imread
from skimage.measure import label

from .layer_manager import LayerManager


class ConnectedComponents(QWidget):
    """Widget to run connected component analysis"""

    def __init__(self, viewer: "napari.viewer.Viewer", label_manager: LayerManager) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        conn_comp_box = QGroupBox("Connected Component Analysis")
        conn_comp_box_layout = QVBoxLayout()

        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self._conn_comp)
        conn_comp_box_layout.addWidget(run_btn)

        conn_comp_box.setLayout(conn_comp_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(conn_comp_box)
        self.setLayout(main_layout)

    def _conn_comp(self):
        """Run connected component analysis to (re) label the labels array"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

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
                scale=self.label_manager.selected_layer.scale
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )
            return True
        else:
            shape = self.label_manager.selected_layer.data.shape
            if len(shape) > 3:
                conn_comp = np.zeros_like(self.label_manager.selected_layer.data)
                for i in range(shape[0]):
                    conn_comp[i] = label(self.label_manager.selected_layer.data[i])

            else:
                conn_comp = label(self.label_manager.selected_layer.data)

            self.label_manager.selected_layer = self.viewer.add_labels(conn_comp,
                name=self.label_manager.selected_layer.name + "_conn_comp",
                scale=self.label_manager.selected_layer.scale
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )
