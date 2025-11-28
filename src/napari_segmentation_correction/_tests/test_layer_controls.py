import numpy as np

from napari_segmentation_correction.layer_control_widgets.layer_manager import (
    LayerManager,
)
from napari_segmentation_correction.layer_controls import LayerControlsWidget


def test_layer_manager(make_napari_viewer):
    viewer = make_napari_viewer()
    label_manager = LayerManager(viewer)
    widget = LayerControlsWidget(viewer, label_manager)

    label_data = np.random.rand(10, 10).astype(np.uint8)
    viewer.add_labels(label_data, name="test_labels")

    labels_layer = viewer.layers["test_labels"]
    viewer.layers.selection.active = labels_layer
    assert widget.label_manager.selected_layer == labels_layer
