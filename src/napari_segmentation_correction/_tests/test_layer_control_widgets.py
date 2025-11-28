import os
from unittest.mock import patch

import dask.array as da
import numpy as np
from napari_builtins.io._read import (
    magic_imread,
)
from qtpy.QtWidgets import (
    QGroupBox,
)

from napari_segmentation_correction.layer_control_widgets import (
    ConvertToNumpyWidget,
    CopyLabelWidget,
    DimensionWidget,
    SaveLabelsWidget,
)
from napari_segmentation_correction.layer_controls import LayerControlsWidget


def test_layer_controls_widget(make_napari_viewer, qtbot):
    """Test initialization of the layer controls widget."""

    viewer = make_napari_viewer()
    widget = LayerControlsWidget(viewer)
    qtbot.addWidget(widget)

    layout = widget.layout()
    assert layout is not None
    assert layout.count() == 5

    # Extract top-level widgets from the main layout
    top_widgets = [layout.itemAt(i).widget() for i in range(layout.count())]

    assert isinstance(top_widgets[0], ConvertToNumpyWidget)
    assert isinstance(top_widgets[1], DimensionWidget)
    assert isinstance(top_widgets[2], QGroupBox)  # plane sliders
    assert isinstance(top_widgets[3], CopyLabelWidget)
    assert isinstance(top_widgets[4], SaveLabelsWidget)


def test_convert_to_numpy(make_napari_viewer, qtbot, img_4d):
    """Test converting a dask array to numpy array"""

    viewer = make_napari_viewer()

    widget = ConvertToNumpyWidget(viewer)
    qtbot.addWidget(widget)

    img = img_4d(dask=True)
    layer = viewer.add_labels(img)
    assert isinstance(viewer.layers[0].data, da.core.Array)
    assert widget.layer == layer
    assert widget.convert_to_array_btn.isEnabled()

    widget._convert_to_array()
    assert isinstance(viewer.layers[0].data, np.ndarray)
    assert not widget.convert_to_array_btn.isEnabled()


def test_dimension_widget(make_napari_viewer, qtbot, img_3d, img_4d):
    """Test changing the dimensions of the layer"""

    viewer = make_napari_viewer()

    widget = DimensionWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.row_active == [False] * 5

    img4d = viewer.add_labels(img_4d())

    assert widget.row_active == [True, True, True, True] + [False]  # first 4 active

    assert "dimension_info" in img4d.metadata

    img3d = viewer.add_labels(img_3d())

    assert widget.row_active == [True, True, True] + [False, False]  # first 3 active

    # Check metadata
    assert "dimension_info" in img3d.metadata
    dims, axes, scale = img3d.metadata["dimension_info"]
    assert len(dims) == 3
    assert len(axes) == 3
    assert len(scale) == 3

    # Change scaling
    for active, (_, _, spin) in zip(
        widget.row_active, widget.axis_widgets, strict=False
    ):
        if active:
            spin.setValue(2.5)

    widget.apply_dims()
    dims, axes, scale = img3d.metadata["dimension_info"]
    assert scale == [2.5, 2.5, 2.5]
    assert tuple(img3d.scale) == (2.5, 2.5, 2.5)

    # change Z -> T, check that scaling for axis 0 is back to 1
    widget.axis_widgets[0][1].setCurrentText("T")
    widget.axis_widgets[1][1].setCurrentText("Y")
    widget.axis_widgets[2][1].setCurrentText("X")
    widget.apply_dims()
    dims, axes, scale = img3d.metadata["dimension_info"]
    assert tuple(scale) == (1, 2.5, 2.5)

    # Transpose dimension order
    viewer.layers.selection.active = img4d
    active_rows = [i for i, a in enumerate(widget.row_active) if a]
    for _, row in zip([0, 1, 2], active_rows, strict=False):
        _, _, spin = widget.axis_widgets[row]

    # Reorder for fun: X,Y,Z
    widget.axis_widgets[0][1].setCurrentText("X")
    widget.axis_widgets[1][1].setCurrentText("Y")
    widget.axis_widgets[2][1].setCurrentText("Z")
    widget.axis_widgets[3][1].setCurrentText("T")

    widget.apply_dims()

    expected_transpose = np.transpose(img_4d(), [3, 2, 1, 0])
    np.testing.assert_array_equal(img4d.data, expected_transpose)

    dims, axes, scale = img4d.metadata["dimension_info"]
    assert axes == ["X", "Y", "Z", "T"]

    # Check that the metadata is loaded when switching back to img3d
    viewer.layers.selection.active = img3d
    assert widget.row_active == [True, True, True] + [False, False]
    assert widget.axis_widgets[0][1].currentText() == "T"


def test_save_labels(make_napari_viewer, qtbot, img_4d, tmp_path):
    """Test saving data with specified dtype and with option to split time points"""

    viewer = make_napari_viewer()

    widget = SaveLabelsWidget(viewer)
    qtbot.addWidget(widget)

    layer = viewer.add_labels(img_4d(dask=True))

    widget.use_compression.setChecked(True)
    widget.select_dtype.setCurrentText("np.uint32")
    widget.filename.setText("Test")

    with patch(
        "napari_segmentation_correction.layer_control_widgets.save_labels_widget.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)
        widget._save_labels()

    result = magic_imread(os.path.join(tmp_path, "Test.tif"))

    np.testing.assert_array_equal(layer.data, result)
    assert result.dtype == np.uint32

    widget.split_time_points.setChecked(True)
    with patch(
        "napari_segmentation_correction.layer_control_widgets.save_labels_widget.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)
        widget._save_labels()

    assert len(os.listdir(tmp_path)) == 4
