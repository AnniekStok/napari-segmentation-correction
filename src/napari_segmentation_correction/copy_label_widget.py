import os

import napari
import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread

from .label_option_layer import LabelOptions
from .layer_manager import LayerManager


class CopyLabelWidget(QWidget):
    """Widget to create a "Labels Options" Layer from which labels can be copied to another layer"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager

        copy_labels_box = QGroupBox("Copy-paste labels")
        copy_labels_layout = QVBoxLayout()

        add_option_layer_btn = QPushButton(
            "Add layer with different label options from folder"
        )
        add_option_layer_btn.clicked.connect(self._add_option_layer)
        convert_to_option_layer_btn = QPushButton(
            "Convert current labels layer to label options layer"
        )
        convert_to_option_layer_btn.clicked.connect(
            self._convert_to_option_layer
        )

        copy_labels_layout.addWidget(add_option_layer_btn)
        copy_labels_layout.addWidget(convert_to_option_layer_btn)
        copy_labels_box.setLayout(copy_labels_layout)

        layout = QVBoxLayout()
        layout.addWidget(copy_labels_box)
        self.setLayout(layout)

    def _add_option_layer(self):
        """Add a new labels layer that contains different alternative segmentations as channels, and add a function to select and copy these cells through shift-clicking"""

        path = QFileDialog.getExistingDirectory(
            self, "Select Label Image Parent Folder"
        )
        if path:
            label_dirs = sorted(
                [
                    d
                    for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))
                ]
            )
            label_stacks = []
            for d in label_dirs:
                # n dirs indicates number of channels
                label_files = sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(path, d))
                        if ".tif" in f
                    ]
                )
                label_imgs = []
                for f in label_files:
                    # n label_files indicates n time points
                    img = imread(os.path.join(path, d, f))
                    label_imgs.append(img)

                if len(label_imgs) > 1:
                    label_stack = np.stack(label_imgs, axis=0)
                    label_stacks.append(label_stack)
                else:
                    label_stacks.append(img)

            if len(label_stacks) > 1:
                self.option_labels = np.stack(label_stacks, axis=0)
            elif len(label_stacks) == 1:
                self.option_labels = label_stacks[0]

            n_channels = len(label_dirs)
            n_timepoints = len(label_files)
            if len(img.shape) == 3:
                n_slices = img.shape[0]
            elif len(img.shape) == 2:
                n_slices = 1

            self.option_labels = self.option_labels.reshape(
                n_channels,
                n_timepoints,
                n_slices,
                img.shape[-2],
                img.shape[-1],
            )

            self.option_labels = np.squeeze(self.option_labels) # squeeze to get rid of dimensions of size 1

        self.option_labels = LabelOptions(
            viewer=self.viewer,
            data=self.option_labels,
            name="label options",
            label_manager=self.label_manager,
        )
        self.viewer.layers.append(self.option_labels)

    def _convert_to_option_layer(self) -> None:

        self.option_labels = LabelOptions(
            viewer=self.viewer,
            data=self.label_manager.selected_layer.data,
            name="label options",
            label_manager=self.label_manager,
        )
        self.viewer.layers.append(self.option_labels)

