import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)
from skimage.util import map_array

from .layer_manager import LayerManager


class ColorFeatureWidget(QWidget):
    """Widget to produce images colored by property"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager

        self.property = QComboBox()
        self.label_manager.layer_update.connect(self.set_properties)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._color_by_feature)

        colorbox = QGroupBox("Color by feature")
        color_layout = QHBoxLayout()

        color_layout.addWidget(self.property)
        color_layout.addWidget(self.run_btn)
        colorbox.setLayout(color_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(colorbox)

        self.setLayout(main_layout)

    def set_properties(self) -> None:
        """Retrieve the available properties and populate the dropdown menu"""

        current_prop = self.property.currentText()
        if self.label_manager.selected_layer is not None:
            props = list(self.label_manager.selected_layer.properties.keys())
            self.property.clear()
            self.property.addItems(
                [p for p in props if p not in ("label", "time_point")]
            )
            if current_prop in props:
                self.property.setCurrentText(current_prop)
            self.run_btn.setEnabled(True) if len(
                props
            ) > 0 else self.run_btn.setEnabled(False)
        else:
            self.run_btn.setEnabled(False)

    def _color_by_feature(self) -> None:
        """Add a new layer to the viewer that displays the labels colored by the selected
        property"""

        feature = self.property.currentText()
        image = map_array(
            self.label_manager.selected_layer.data,
            self.label_manager.selected_layer.properties["label"],
            self.label_manager.selected_layer.properties[feature],
        ).astype(np.float32)
        self.viewer.add_image(
            image, colormap="turbo", scale=self.label_manager.selected_layer.scale
        )
