import dask.array as da
import napari
import numpy as np
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
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

# List of available integer types in order
_int_types = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
]


def _minimal_safe_dtype(
    img1: np.ndarray, img2: np.ndarray, op: str = "sub"
) -> np.dtype:
    """Return the minimal safe dtype for a given operation between between two arrays"""

    # If either is float, promote to larger float
    if np.issubdtype(img1.dtype, np.floating) or np.issubdtype(img2.dtype, np.floating):
        return np.result_type(img1, img2)

    # Get ranges for inputs
    info1 = np.iinfo(img1.dtype)
    info2 = np.iinfo(img2.dtype)

    if op == "add":
        min_val = int(info1.min) + int(info2.min)
        max_val = int(info1.max) + int(info2.max)
    elif op == "sub":
        min_val = int(info1.min) - int(info2.max)
        max_val = int(info1.max) - int(info2.min)
    elif op == "mul":
        # consider all combinations
        candidates = [
            int(info1.min) * int(info2.min),
            int(info1.min) * int(info2.max),
            int(info1.max) * int(info2.min),
            int(info1.max) * int(info2.max),
        ]
        min_val, max_val = int(min(candidates)), int(max(candidates))
    elif op == "div":
        # division → float64 is safest
        return np.float64
    else:
        min_val, max_val = int(info1.min), int(info1.max)

    # Pick the smallest integer type that fits [min_val, max_val]
    for dtype in _int_types:
        dtype_info = np.iinfo(dtype)
        if min_val >= dtype_info.min and max_val <= dtype_info.max:
            return dtype

    return np.int64


def _adjust_or_clip(
    img: np.ndarray, original_dtype: np.dtype, adjust_dtype: bool
) -> np.ndarray:
    """Return the image in a new dtype or clip it to the original dtype."""

    if adjust_dtype:
        # If original was integer, return the smallest integer dtype that fits the img
        if np.issubdtype(original_dtype, np.integer):
            rmin = int(img.min())
            rmax = int(img.max())
            for dtype in _int_types:
                info = np.iinfo(dtype)
                if rmin >= info.min and rmax <= info.max:
                    return img.astype(dtype)
            return img.astype(np.int64)
        else:
            # For floats, choose a float type that can represent the img
            return img.astype(np.result_type(img.dtype, original_dtype))
    else:
        # Clip to the bounds of original dtype
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            img = np.clip(img, info.min, info.max)
            return img.astype(original_dtype)
        else:
            # For floats, no need to clip
            return img


def add_images(img1: np.ndarray, img2: np.ndarray, adjust_dtype=True) -> np.ndarray:
    """Add img2 to img1"""

    original_dtype = img1.dtype
    safe_dtype = _minimal_safe_dtype(img1, img2, op="add")
    result = img1.astype(safe_dtype) + img2.astype(safe_dtype)
    return _adjust_or_clip(result, original_dtype, adjust_dtype)


def subtract_images(
    img1: np.ndarray, img2: np.ndarray, adjust_dtype=True
) -> np.ndarray:
    """Subtract img2 from img1"""

    original_dtype = img1.dtype
    safe_dtype = _minimal_safe_dtype(img1, img2, op="sub")
    result = img1.astype(safe_dtype) - img2.astype(safe_dtype)
    return _adjust_or_clip(result, original_dtype, adjust_dtype)


def multiply_images(
    img1: np.ndarray, img2: np.ndarray, adjust_dtype=True
) -> np.ndarray:
    """Multiply img1 by img2"""

    original_dtype = img1.dtype
    safe_dtype = _minimal_safe_dtype(img1, img2, op="mul")
    result = img1.astype(safe_dtype) * img2.astype(safe_dtype)
    return _adjust_or_clip(result, original_dtype, adjust_dtype)


def divide_images(img1: np.ndarray, img2: np.ndarray, adjust_dtype=True) -> np.ndarray:
    """Divide img1 by img2"""

    original_dtype = img1.dtype
    # Division always promotes to float to avoid truncation
    result = np.divide(
        img1, img2, out=np.zeros_like(img1, dtype=np.float64), where=img2 != 0
    )
    if adjust_dtype:
        return result  # always float64 for division
    else:
        if np.issubdtype(original_dtype, np.integer):
            # Clip and convert back to original dtype
            info = np.iinfo(original_dtype)
            result = np.clip(result, info.min, info.max)
            return result.astype(original_dtype)
        else:
            return result


def logical_and(img1: np.ndarray, img2: np.ndarray, **kwargs) -> np.ndarray:
    """Compute logical AND of img1 and img2"""

    return np.logical_and(img1 != 0, img2 != 0).astype(int)


def logical_or(img1: np.ndarray, img2: np.ndarray, **kwargs) -> np.ndarray:
    """Compute logical OR of img1 and img2"""
    return np.logical_or(img1 != 0, img2 != 0).astype(int)


class ImageCalculator(QWidget):
    """Widget to perform calculations between two images"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()

        self.viewer = viewer

        box = QGroupBox("Image Calculator")
        box_layout = QVBoxLayout()

        image1_layout = QHBoxLayout()
        image1_layout.addWidget(QLabel("Label image 1"))
        self.image1_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image1_dropdown.layer_changed.connect(self._update_image1)
        image1_layout.addWidget(self.image1_dropdown)
        box_layout.addLayout(image1_layout)

        image2_layout = QHBoxLayout()
        image2_layout.addWidget(QLabel("Label image 2"))
        self.image2_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image2_dropdown.layer_changed.connect(self._update_image2)
        image2_layout.addWidget(self.image2_dropdown)
        box_layout.addLayout(image2_layout)

        operation_layout = QHBoxLayout()
        self.operation = QComboBox()
        self.operation.addItem("Add")
        self.operation.addItem("Subtract")
        self.operation.addItem("Multiply")
        self.operation.addItem("Divide")
        self.operation.addItem("AND")
        self.operation.addItem("OR")
        operation_layout.addWidget(QLabel("Operation"))
        operation_layout.addWidget(self.operation)
        box_layout.addLayout(operation_layout)

        self.maintain_dtype = QCheckBox("Keep original data type")
        self.maintain_dtype.setToolTip(
            "If checked, the result will retain the original data type and hold zeroes "
            "where the pixels would have had negative values. Otherwise, the data type "
            "will be updated if necessary to accomodate negative values."
        )
        box_layout.addWidget(self.maintain_dtype)

        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self._calculate_images)
        run_btn.setEnabled(
            self.image1_dropdown.selected_layer is not None
            and self.image2_dropdown.selected_layer is not None
        )
        self.image1_dropdown.layer_changed.connect(
            lambda: run_btn.setEnabled(
                self.image1_dropdown.selected_layer is not None
                and self.image2_dropdown.selected_layer is not None
            )
        )
        self.image2_dropdown.layer_changed.connect(
            lambda: run_btn.setEnabled(
                self.image1_dropdown.selected_layer is not None
                and self.image2_dropdown.selected_layer is not None
            )
        )

        box_layout.addWidget(run_btn)
        box.setLayout(box_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(box)
        self.setLayout(main_layout)

    def _update_image1(self, selected_layer: str) -> None:
        """Update the layer for image 1"""

        if selected_layer == "":
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

    def _update_image2(self, selected_layer: str) -> None:
        """Update the layer for image 2"""

        if selected_layer == "":
            self.image2_layer = None
        else:
            self.image2_layer = self.viewer.layers[selected_layer]
            self.image2_dropdown.setCurrentText(selected_layer)

    def _calculate_images(self) -> None:
        """Execute mathematical operations between two images."""

        if self.image1_layer.data.shape != self.image2_layer.data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Images must have the same shape")
            msg.setText("Images must have the same shape")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        if self.operation.currentText() == "Add":
            action = add_images
        elif self.operation.currentText() == "Subtract":
            action = subtract_images
        elif self.operation.currentText() == "Multiply":
            action = multiply_images
        elif self.operation.currentText() == "Divide":
            action = divide_images
        elif self.operation.currentText() == "AND":
            action = logical_and
        elif self.operation.currentText() == "OR":
            action = logical_or

        adjust_dtype = not self.maintain_dtype.isChecked()

        if isinstance(self.image1_layer.data, da.core.Array) or isinstance(
            self.image2_layer.data, da.core.Array
        ):
            indices = range(self.image1_layer.data.shape[0])
            result = process_action(
                self.image1_layer.data,
                self.image2_layer.data,
                action,
                basename=self.image1_layer.name,
                img1_index=indices,
                img2_index=indices,
                adjust_dtype=adjust_dtype,
            )
        else:
            result = process_action(
                self.image1_layer.data,
                self.image2_layer.data,
                action,
                basename=self.image1_layer.name,
                adjust_dtype=adjust_dtype,
            )

        if result is not None:
            if np.issubdtype(result.dtype, np.integer) and (
                isinstance(self.image1_layer, napari.layers.Labels)
                and isinstance(self.image2_layer, napari.layers.Labels)
            ):
                self.viewer.add_labels(
                    result,
                    name=f"{self.image1_layer.name}_{self.image2_layer.name}_{self.operation.currentText()}",
                    scale=self.image1_layer.scale,
                )
            else:
                self.viewer.add_image(
                    result,
                    name=f"{self.image1_layer.name}_{self.image2_layer.name}_{self.operation.currentText()}",
                    scale=self.image1_layer.scale,
                )
