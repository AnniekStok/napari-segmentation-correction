from typing import Tuple

import napari
from PyQt5.QtCore import pyqtSignal
from qtpy.QtWidgets import QComboBox
from ._layer_dropdown import LayerDropdown
import dask.array as da
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

class LayerManager(QWidget):
    """QComboBox widget with functions for updating the selected layer and to update the list of options when the list of layers is modified."""

    
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer
        self._selected_layer = None
        
        self.label_dropdown = LayerDropdown(self.viewer, (napari.layers.Labels))
        self.label_dropdown.layer_changed.connect(self._update_labels)

        layout = QVBoxLayout()
        layout.addWidget(self.label_dropdown)

        self.setLayout(layout)
        
    @property
    def selected_layer(self):
        return self._selected_layer

    def _update_labels(self, selected_layer) -> None:
        """Update the layer that is set to be the 'labels' layer that is being edited."""

        if selected_layer == "":
            self._selected_layer = None
        else:
            self._selected_layer = self.viewer.layers[selected_layer]
            self.label_dropdown.setCurrentText(selected_layer)
            # self.convert_to_array_btn.setEnabled(
            #     isinstance(self.labels.data, da.core.Array)
            # )
            print('new labels', self._selected_layer)