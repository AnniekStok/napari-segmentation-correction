from napari_segmentation_toolbox.layer_controls import LayerControlsWidget
from napari_segmentation_toolbox.plot_widget import PlotWidget
from napari_segmentation_toolbox.regionprops_widget import RegionPropsWidget


def test_regionprops_widget(make_napari_viewer, qtbot, img_3d):
    """Test Regionprops widget."""

    viewer = make_napari_viewer()
    regionprops_widget = RegionPropsWidget(viewer)
    qtbot.addWidget(regionprops_widget)

    layer_controls = LayerControlsWidget(viewer)
    qtbot.addWidget(layer_controls)

    plot_widget = PlotWidget(viewer)
    qtbot.addWidget(plot_widget)

    layer_controls.dimension_widget.update_status.connect(
        regionprops_widget.update_properties_and_callback
    )

    layer = viewer.add_labels(img_3d())
    assert len(regionprops_widget.checkboxes) == 11

    for ch in regionprops_widget.checkboxes:
        if ch["region_prop_name"] == "volume":
            assert ch["checkbox"].isEnabled()
            ch["checkbox"].setChecked(True)
        elif ch["region_prop_name"] == "area":
            assert not ch["checkbox"].isEnabled()

    assert regionprops_widget.table is None
    regionprops_widget._measure()
    assert regionprops_widget.table is not None
    assert plot_widget.x_combo.currentText() == "label"
    assert plot_widget.y_combo.currentText() == "volume"

    regionprops_widget.color_by_feature_widget.property.setCurrentText("volume")
    regionprops_widget.color_by_feature_widget._color_by_feature()
    assert len(viewer.layers) == 2

    viewer.layers.selection.active = layer
    regionprops_widget.prop_filter_widget.property.setCurrentText("volume")
    regionprops_widget.prop_filter_widget.value.setValue(150)
    regionprops_widget.prop_filter_widget.filter_by_property()
    assert len(viewer.layers) == 3
    assert regionprops_widget.prop_filter_widget.layer.data[2, 8, 5] == 0
