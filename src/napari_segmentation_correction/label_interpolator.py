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
    QLabel,
    QComboBox
)
from skimage.io import imread
from .layer_manager import LayerManager
from scipy.ndimage import distance_transform_edt


import numpy as np
from scipy.ndimage import distance_transform_edt

def signed_distance_transform(mask):
    mask = mask.astype(bool)
    dist_out = distance_transform_edt(~mask)
    dist_in = distance_transform_edt(mask)
    return dist_out - dist_in

def interpolate_binary_mask(mask):
    """
    Interpolates a sparse binary mask array using SDTs along the first axis.
    
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


class InterpolationWidget(QWidget):
    """Widget to interpolate between nonzero pixels in a label layer using signed distance transforms."""

    def __init__(self, viewer: "napari.viewer.Viewer", label_manager: LayerManager) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        interpolator_box = QGroupBox("Interpolate mask")
        interpolator_box_layout = QVBoxLayout()
 
        run_btn = QPushButton("Run interpolation along first axis")
        run_btn.clicked.connect(self._interpolate)
        interpolator_box_layout.addWidget(run_btn)

        interpolator_box.setLayout(interpolator_box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(interpolator_box)
        self.setLayout(main_layout)

    def _interpolate(self):
        """Interpolate between the nonzero pixels in the selected layer"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_interpolated"),
            )
            if os.path.exists(outputdir):
                shutil.rmtree(outputdir)
            os.mkdir(outputdir)

            in_memory_stack = []
            for i in range(
                self.label_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                current_stack = self.label_manager.selected_layer.data[
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
                            self.label_manager.selected_layer.name
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
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name + "_interpolated",
                scale=self.label_manager.selected_layer.scale
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )
            return True
        else:
            interpolated = interpolate_binary_mask(self.label_manager.selected_layer.data)   

            self.label_manager.selected_layer = self.viewer.add_labels(interpolated,
                name=self.label_manager.selected_layer.name + "_interpolated",
                scale=self.label_manager.selected_layer.scale
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )
