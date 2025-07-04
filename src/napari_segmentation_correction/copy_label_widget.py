import napari
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

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

        convert_to_option_layer_btn = QPushButton(
            "Convert current labels layer to label options layer"
        )
        convert_to_option_layer_btn.clicked.connect(
            self._convert_to_option_layer
        )

        copy_labels_layout.addWidget(convert_to_option_layer_btn)
        copy_labels_box.setLayout(copy_labels_layout)

        layout = QVBoxLayout()
        layout.addWidget(copy_labels_box)
        self.setLayout(layout)

    def _convert_to_option_layer(self) -> None:

        self.option_labels = LabelOptions(
            viewer=self.viewer,
            data=self.label_manager.selected_layer.data,
            name="label options",
            label_manager=self.label_manager,
        )
        self.viewer.layers.append(self.option_labels)

