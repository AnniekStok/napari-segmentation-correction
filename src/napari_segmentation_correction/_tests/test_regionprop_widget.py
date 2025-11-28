from napari_segmentation_correction.main_widget import LabelToolbox


def test_regionprops_widget(make_napari_viewer, qtbot, img_3d):
    """Test Regionprops widget."""

    viewer = make_napari_viewer()
    main_widget = LabelToolbox(viewer)
    qtbot.addWidget(main_widget)

    layer = viewer.add_labels(img_3d())
    assert len(main_widget.regionprops_widget.checkboxes) == 11
    assert main_widget.regionprops_widget.feature_dims == 3

    for ch in main_widget.regionprops_widget.checkboxes:
        if ch["region_prop_name"] == "volume":
            assert ch["checkbox"].isEnabled()
            ch["checkbox"].setChecked(True)
        elif ch["region_prop_name"] == "area":
            assert not ch["checkbox"].isEnabled()

    assert main_widget.regionprops_widget.table is None
    main_widget.regionprops_widget._measure()
    assert main_widget.regionprops_widget.table is not None
    assert main_widget.plot_widget.x_combo.currentText() == "label"
    assert main_widget.plot_widget.y_combo.currentText() == "volume"

    main_widget.regionprops_widget.color_by_feature_widget.property.setCurrentText(
        "volume"
    )
    main_widget.regionprops_widget.color_by_feature_widget._color_by_feature()
    assert len(viewer.layers) == 2

    viewer.layers.selection.active = layer
    main_widget.regionprops_widget.prop_filter_widget.property.setCurrentText("volume")
    main_widget.regionprops_widget.prop_filter_widget.value.setValue(150)
    main_widget.regionprops_widget.prop_filter_widget.filter_by_property()
    assert len(viewer.layers) == 3
    assert main_widget.regionprops_widget.prop_filter_widget.layer.data[2, 8, 5] == 0
