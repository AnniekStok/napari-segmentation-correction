from unittest.mock import MagicMock

import numpy as np
import pytest
from napari.utils import DirectLabelColormap
from qtpy.QtCore import Qt

from napari_segmentation_toolbox.layer_controls import LayerControlsWidget
from napari_segmentation_toolbox.plot_widget import PlotWidget
from napari_segmentation_toolbox.regionprops_widget import RegionPropsWidget


@pytest.fixture
def setup_props_widget(make_napari_viewer, qtbot, img_4d):
    """Creates viewer + RegionPropsWidget + layer, returns widget."""
    viewer = make_napari_viewer()

    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    layer_controls = LayerControlsWidget(viewer)
    qtbot.addWidget(layer_controls)
    plot_widget = PlotWidget(viewer)
    qtbot.addWidget(plot_widget)

    layer_controls.dimension_widget.update_status.connect(
        widget.update_properties_and_callback
    )

    layer = viewer.add_labels(img_4d())

    # force computation so table exists
    widget._measure()

    return widget.table, viewer, layer


# Single row selection
def test_table_single_selection(setup_props_widget, qtbot):
    widget, viewer, layer = setup_props_widget

    table = widget._table_widget
    widget.special_selection = [4]

    original = widget._update_label_colormap
    widget._update_label_colormap = MagicMock(side_effect=original)

    # click first row, column 0
    idx = table.model().index(0, 0)
    qtbot.mouseClick(
        table.viewport(), Qt.LeftButton, pos=table.visualRect(idx).center()
    )

    assert len(table.selectedIndexes()) > 0
    assert widget.special_selection == []
    qtbot.waitUntil(lambda: widget._update_label_colormap.called, timeout=1000)
    assert isinstance(layer.colormap, DirectLabelColormap)

    assert float(layer.colormap.color_dict[1][-1]) == 1.0
    np.testing.assert_allclose(layer.colormap.color_dict[2][-1], 0.6, rtol=1e-6)


# Right-click → update special_selection
def test_table_right_click_special_selection(setup_props_widget, qtbot):
    widget, viewer, layer = setup_props_widget

    table = widget._table_widget

    original = widget._update_label_colormap
    widget._update_label_colormap = MagicMock(side_effect=original)

    row2 = 2
    row4 = 4

    # simulate right-click without qtbot
    widget._clicked_table(right=True, shift=False, index=table.model().index(row2, 0))
    qtbot.waitUntil(lambda: widget._update_label_colormap.called, timeout=1000)
    assert isinstance(layer.colormap, DirectLabelColormap)
    assert widget.special_selection == [widget._table["label"][row2]]
    assert float(layer.colormap.color_dict[0][-1]) == 0.0
    assert float(layer.colormap.color_dict[2][-1]) == 1.0

    widget._clicked_table(right=True, shift=True, index=table.model().index(row4, 0))

    qtbot.waitUntil(
        lambda: float(layer.colormap.color_dict[row4][-1]) == 1.0, timeout=1000
    )

    # now safe to assert
    assert widget.special_selection == [
        widget._table["label"][row2],
        widget._table["label"][row4],
    ]
