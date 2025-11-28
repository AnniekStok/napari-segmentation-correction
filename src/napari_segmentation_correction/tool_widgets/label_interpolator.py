import os

import dask.array as da
import napari
import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
)
from scipy.ndimage import distance_transform_edt
from skimage.io import imread

from napari_segmentation_correction.helpers.base_tool_widget import BaseToolWidget
from napari_segmentation_correction.helpers.process_actions_helpers import (
    remove_invalid_chars,
)


def signed_distance_transform(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    dist_out = distance_transform_edt(~mask)
    dist_in = distance_transform_edt(mask)
    return dist_out - dist_in


def interpolate_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Interpolates a binary mask array using SDTs along the first axis.
    Args:
        mask (ndarray): Binary array of shape (T, X, Y, Z) or (X, Y, Z), etc.,
                        where some slices along 'axis' contain valid masks
    Returns:
        ndarray: Binary array of same shape with interpolated masks along 'axis'
    """

    output = np.zeros_like(mask, dtype=np.uint8)

    # Find slices along axis that have any nonzero values
    valid_idxs = [i for i in range(mask.shape[0]) if np.any(mask[i])]

    for i in range(len(valid_idxs) - 1):
        i_start, i_end = valid_idxs[i], valid_idxs[i + 1]
        sdt_start = signed_distance_transform(mask[i_start])
        sdt_end = signed_distance_transform(mask[i_end])

        for j in range(i_start, i_end + 1):
            alpha = (j - i_start) / (i_end - i_start)
            sdt_interp = (1 - alpha) * sdt_start + alpha * sdt_end
            output[j] = (sdt_interp < 0).astype(np.uint8)

    return output


class InterpolationWidget(BaseToolWidget):
    """Widget to interpolate between nonzero pixels in a label layer using signed
    distance transforms."""

    def __init__(
        self, viewer: "napari.viewer.Viewer", layer_type=(napari.layers.Labels)
    ) -> None:
        super().__init__(viewer, layer_type)

        interpolator_box = QGroupBox("Interpolate mask")
        interpolator_box_layout = QVBoxLayout()

        run_btn = QPushButton("Run interpolation along first axis")
        run_btn.clicked.connect(self._interpolate)
        run_btn.setEnabled(self.layer is not None)
        self.update_status.connect(lambda: run_btn.setEnabled(self.layer is not None))
        interpolator_box_layout.addWidget(run_btn)

        interpolator_box.setLayout(interpolator_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(interpolator_box)
        self.setLayout(main_layout)

    def _interpolate(self) -> None:
        """Interpolate between the nonzero pixels in the selected layer"""

        if isinstance(self.layer.data, da.core.Array):
            outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if not outputdir:
                return

            outputdir = os.path.join(
                outputdir,
                remove_invalid_chars(self.layer.name + "_interpolated"),
            )
            while os.path.exists(outputdir):
                outputdir = outputdir + "_1"
            os.mkdir(outputdir)

            in_memory_stack = []
            for i in range(self.layer.data.shape[0]):  # Loop over the first dimension
                current_stack = self.layer.data[
                    i
                ].compute()  # Compute the current stack

                in_memory_stack.append(current_stack)

            in_memory_stack = np.stack(in_memory_stack, axis=0)
            interpolated = interpolate_binary_mask(in_memory_stack)

            for i in range(interpolated.shape[0]):
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.layer.name
                            + "_interpolation_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(interpolated[i], dtype="uint8"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.layer.name + "_interpolated",
                scale=self.layer.scale,
            )
        else:
            interpolated = interpolate_binary_mask(self.layer.data)

            self.layer = self.viewer.add_labels(
                interpolated,
                name=self.layer.name + "_interpolated",
                scale=self.layer.scale,
            )
