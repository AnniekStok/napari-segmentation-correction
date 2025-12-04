import os
from unittest.mock import patch

import dask.array as da
import numpy as np
from napari_builtins.io._read import (
    magic_imread,
)

from napari_segmentation_correction.layer_control_widgets import (
    ConvertToNumpyWidget,
    CopyLabelWidget,
    DimensionWidget,
    SaveLabelsWidget,
)
from napari_segmentation_correction.layer_controls import LayerControlsWidget
from napari_segmentation_correction.plane_slider_widget import PlaneSliderWidget


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
    assert isinstance(top_widgets[2], PlaneSliderWidget)  # plane sliders
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


def test_dimension_widget(make_napari_viewer, qtbot, img_2d, img_3d, img_4d):
    """Test changing the dimensions of the layer"""

    viewer = make_napari_viewer()

    widget = DimensionWidget(viewer)
    qtbot.addWidget(widget)

    assert not widget.pos_combos[0].isEnabled()
    assert not widget.pos_combos[1].isEnabled()
    assert not widget.pos_combos[2].isEnabled()
    assert not widget.pos_combos[3].isEnabled()
    assert not widget.pos_combos[4].isEnabled()

    img2d = viewer.add_labels(img_2d)
    assert "dimensions" in img2d.metadata

    assert widget.pos_combos[0].isEnabled()
    assert widget.pos_combos[1].isEnabled()
    assert not widget.pos_combos[2].isEnabled()
    assert not widget.pos_combos[3].isEnabled()
    assert not widget.pos_combos[4].isEnabled()

    img3d = viewer.add_labels(img_3d())
    assert "dimensions" in img3d.metadata
    dims = img3d.metadata["dimensions"]
    assert len(dims) == 3

    assert widget.pos_combos[0].isEnabled()
    assert widget.pos_combos[1].isEnabled()
    assert widget.pos_combos[2].isEnabled()
    assert not widget.pos_combos[3].isEnabled()
    assert not widget.pos_combos[4].isEnabled()

    # Change scaling
    widget.scale_widgets[0].setValue(2.5)
    widget.scale_widgets[1].setValue(2.5)
    widget.scale_widgets[2].setValue(2.5)
    widget._apply_scale_single()

    assert tuple(img3d.scale) == (2.5, 2.5, 2.5)

    # change Z -> T
    widget.name_widgets[0].setCurrentText("T")
    widget.name_widgets[1].setCurrentText("Y")
    widget.name_widgets[2].setCurrentText("X")
    widget._apply_names()
    dims = img3d.metadata["dimensions"]
    assert tuple(dims) == ("T", "Y", "X")

    # Transpose dimension order
    img4d = viewer.add_labels(img_4d())
    assert "dimensions" in img4d.metadata

    assert widget.pos_combos[0].isEnabled()
    assert widget.pos_combos[1].isEnabled()
    assert widget.pos_combos[2].isEnabled()
    assert widget.pos_combos[3].isEnabled()
    assert not widget.pos_combos[4].isEnabled()

    widget.pos_combos[0].setCurrentText("1")
    widget.pos_combos[1].setCurrentText("0")
    widget.pos_combos[2].setCurrentText("2")
    widget.pos_combos[3].setCurrentText("3")
    widget._apply_axis_reorder()

    expected_transpose = np.transpose(img_4d(), [1, 0, 2, 3])
    np.testing.assert_array_equal(img4d.data, expected_transpose)

    # test that invalid reorder does not change the data
    widget.pos_combos[0].setCurrentText("1")
    widget.pos_combos[1].setCurrentText("1")
    widget.pos_combos[2].setCurrentText("2")
    widget.pos_combos[3].setCurrentText("3")
    widget._apply_axis_reorder()

    np.testing.assert_array_equal(img4d.data, expected_transpose)  # data unchanged

    # test that invalid change of names does not trigger update
    dims = img4d.metadata["dimensions"]
    assert tuple(dims) == ("T", "Z", "Y", "X")  # default names
    widget.name_widgets[0].setCurrentText("T")
    widget.name_widgets[1].setCurrentText("T")
    widget.name_widgets[2].setCurrentText("Y")
    widget.name_widgets[3].setCurrentText("X")
    widget._apply_names()
    dims = img4d.metadata["dimensions"]
    assert tuple(dims) == ("T", "Z", "Y", "X")  # still the original names

    # Check that the metadata is loaded when switching back to img3d
    viewer.layers.selection.active = img3d

    assert widget.pos_combos[0].isEnabled()
    assert widget.pos_combos[1].isEnabled()
    assert widget.pos_combos[2].isEnabled()
    assert not widget.pos_combos[3].isEnabled()
    assert not widget.pos_combos[4].isEnabled()

    assert widget.name_widgets[0].currentText() == "T"


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
