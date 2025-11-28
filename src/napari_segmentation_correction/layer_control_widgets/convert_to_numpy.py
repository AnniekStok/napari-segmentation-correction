import dask.array as da
import napari
import numpy as np
from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
)

from napari_segmentation_correction.helpers.base_tool_widget import BaseToolWidget


class ConvertToNumpyWidget(BaseToolWidget):
    """Widget to convert current layer from dask array to in memory numpy array."""

    def __init__(
        self,
        viewer: napari.Viewer,
        layer_type=(napari.layers.Labels, napari.layers.Labels),
    ):
        super().__init__(viewer, layer_type)

        ### Add option to convert dask array to in-memory array
        self.convert_to_array_btn = QPushButton("Convert to in-memory array")
        self.convert_to_array_btn.setEnabled(
            self.layer is not None and isinstance(self.layer.data, da.core.Array)
        )
        self.convert_to_array_btn.clicked.connect(self._convert_to_array)

        layout = QVBoxLayout()
        layout.addWidget(self.convert_to_array_btn)
        self.setLayout(layout)

    def _convert_to_array(self) -> None:
        """Convert from dask array to in-memory array. This is necessary for manual
        editing using the label tools (brush, eraser, fill bucket)."""

        if isinstance(self.layer.data, da.core.Array):
            stack = []
            for i in range(self.layer.data.shape[0]):
                current_stack = self.layer.data[i].compute()
                stack.append(current_stack)
            self.layer.data = np.stack(stack, axis=0)
            self.update_status.emit()
            self.convert_to_array_btn.setEnabled(False)
