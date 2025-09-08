import dask.array as da
import napari
import pandas as pd
from napari.layers import Labels
from qtpy.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from skimage import measure

from .custom_table_widget import ColoredTableWidget
from .layer_manager import LayerManager


class RegionPropsWidget(QWidget):
    """Widget showing region props as a table and plot widget"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.table = None

        self.table_btn = QPushButton("Compute properties")
        self.table_btn.clicked.connect(self._create_summary_table)
        self.table_btn.setEnabled(isinstance(self.label_manager.selected_layer, Labels))
        self.label_manager.layer_update.connect(
            lambda: self.table_btn.setEnabled(
                isinstance(self.label_manager.selected_layer, napari.layers.Labels)
            )
        )
        self.regionprops_layout = QVBoxLayout()
        self.regionprops_layout.addWidget(self.table_btn)
        self.regionprops_layout.addStretch()
        self.setLayout(self.regionprops_layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _create_summary_table(self) -> None:
        """Create table displaying the sizes of the different labels in the current stack"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            props_list = []
            for tp in range(self.label_manager.selected_layer.data.shape[0]):
                props = measure.regionprops_table(
                    self.label_manager.selected_layer.data[tp].compute(),
                    properties=["label", "num_pixels", "area", "centroid"],
                    spacing=self.label_manager.selected_layer.scale[1:],
                )
                props["time_point"] = tp
                props_list.append(pd.DataFrame.from_dict(props))

            props = pd.concat(props_list)

            if hasattr(self.label_manager.selected_layer, "properties"):
                self.label_manager.selected_layer.properties = props

        else:
            if len(self.label_manager.selected_layer.data.shape) == 4:
                props_list = []
                for tp in range(self.label_manager.selected_layer.data.shape[0]):
                    props = measure.regionprops_table(
                        self.label_manager.selected_layer.data[tp],
                        properties=["label", "num_pixels", "area", "centroid"],
                        spacing=self.label_manager.selected_layer.scale[1:],
                    )
                    props["time_point"] = tp
                    props_list.append(pd.DataFrame.from_dict(props))

                props = pd.concat(props_list)

                if hasattr(self.label_manager.selected_layer, "properties"):
                    self.label_manager.selected_layer.properties = props

            elif len(self.label_manager.selected_layer.data.shape) in (2, 3):
                props = measure.regionprops_table(
                    self.label_manager.selected_layer.data,
                    properties=["label", "num_pixels", "area", "centroid"],
                    spacing=self.label_manager.selected_layer.scale,
                )
                if hasattr(self.label_manager.selected_layer, "properties"):
                    self.label_manager.selected_layer.properties = (
                        pd.DataFrame.from_dict(props)
                    )
            else:
                print("input should be a 2D, 3D or 4D array")
                self.table = None
                if self.table is not None:
                    self.table.hide()
                return

        # add the napari-skimage-regionprops inspired table to the viewer
        if self.table is not None:
            self.table.hide()

        if self.viewer is not None:
            self.table = ColoredTableWidget(
                self.label_manager.selected_layer, self.viewer
            )
            self.table._set_label_colors_to_rows()
            self.table.setMinimumWidth(500)
            self.regionprops_layout.addWidget(self.table)
