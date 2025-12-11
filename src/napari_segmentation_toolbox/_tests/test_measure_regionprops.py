import numpy as np
from qtpy.QtWidgets import QMessageBox

from napari_segmentation_toolbox.regionprops_widget import RegionPropsWidget


def make_labels(shape):
    arr = np.zeros(shape, dtype=np.uint32)
    # put a single blob in every T/C slice or volume
    it = np.nditer(np.zeros(shape[:-2]), flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        sl = idx + (slice(10, 15), slice(10, 15))
        arr[sl] = 1
    return arr


def make_intensity(shape):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def add_metadata(layer, dims):
    layer.metadata["dimensions"] = dims


def run_measure(
    viewer, widget, labels_data, label_dims, intensity_data, intensity_dims
):
    """Utility to create layers, assign metadata, and run _measure."""
    # add layers
    lbl_layer = viewer.add_labels(labels_data)
    add_metadata(lbl_layer, label_dims)

    if intensity_data is not None:
        int_layer = viewer.add_image(intensity_data)
        add_metadata(int_layer, intensity_dims)
        widget.intensity_image_dropdown.selected_layer = int_layer
    else:
        widget.intensity_image_dropdown.selected_layer = None

    # spacing = 1
    lbl_layer.scale = [1] * len(label_dims)

    # Force properties = ['intensity_mean']
    def fake_get_selected_features():
        return ["intensity_mean"]

    widget._get_selected_features = fake_get_selected_features

    widget.layer = lbl_layer

    widget._measure()
    return widget.layer.properties


def test_measure_yx_yx(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    labels = make_labels((100, 100))
    intensity = make_intensity((100, 100))

    props = run_measure(
        viewer,
        widget,
        labels_data=labels,
        label_dims=["Y", "X"],
        intensity_data=intensity,
        intensity_dims=["Y", "X"],
    )

    assert "intensity_mean" in props
    assert len(props) == 2


def test_measure_yx_cyx(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    labels = make_labels((100, 100))
    intensity = make_intensity((3, 100, 100))

    props = run_measure(
        viewer,
        widget,
        labels_data=labels,
        label_dims=["Y", "X"],
        intensity_data=intensity,
        intensity_dims=["C", "Y", "X"],
    )

    assert "intensity_mean-1" in props
    assert len(props) == 4


def test_measure_zyx_zcyx(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    labels = make_labels((30, 100, 100))
    intensity = make_intensity((30, 2, 100, 100))

    props = run_measure(
        viewer,
        widget,
        labels_data=labels,
        label_dims=["Z", "Y", "X"],
        intensity_data=intensity,
        intensity_dims=["Z", "C", "Y", "X"],
    )

    assert "intensity_mean-1" in props
    assert len(props) == 3


def test_measure_tyx_tyx(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    labels = make_labels((2, 80, 80))  # T, Y, X
    intensity = make_intensity((2, 80, 80))

    props = run_measure(
        viewer,
        widget,
        labels_data=labels,
        label_dims=["T", "Y", "X"],
        intensity_data=intensity,
        intensity_dims=["T", "Y", "X"],
    )

    assert len(props) == 3
    assert "time_point" in props
    assert sorted(props["time_point"]) == [0, 1]


def test_measure_cyx_yxc(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    labels = make_labels((3, 80, 80))  # C, Y, X
    intensity = make_intensity((80, 80, 4))  # intensity channels only

    props = run_measure(
        viewer,
        widget,
        labels_data=labels,
        label_dims=["C", "Y", "X"],
        intensity_data=intensity,
        intensity_dims=["Y", "X", "C"],
    )

    # must compute 1 entry per label channel
    assert len(props) == 6  # 4 intensity measurements, 1 label, 1 channel
    assert "intensity_mean-1" in props
    assert "channel" in props
    assert sorted(props["channel"]) == [0, 1, 2]


def test_measure_tzcyx_tzyx(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    labels = make_labels((2, 3, 2, 80, 80))  # TZCYX
    intensity = make_intensity((2, 3, 80, 80))  # TZYX

    props = run_measure(
        viewer,
        widget,
        labels_data=labels,
        label_dims=["T", "Z", "C", "Y", "X"],
        intensity_data=intensity,
        intensity_dims=["T", "Z", "Y", "X"],
    )

    assert len(props) == 4  # time_point, label, channel, intensity_mean
    assert "time_point" in props
    assert "channel" in props
    assert "intensity_mean" in props
    assert sorted(props["time_point"]) == [0, 0, 1, 1]


def test_measure_invalid_shape_failure(make_napari_viewer, qtbot, monkeypatch):
    viewer = make_napari_viewer()
    widget = RegionPropsWidget(viewer)
    qtbot.addWidget(widget)

    called = {}

    def fake_exec_(self):
        called["title"] = self.windowTitle()
        called["text"] = self.text()
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "exec_", fake_exec_)

    labels = make_labels((50, 50))
    intensity = make_intensity((30, 50, 3))  # mismatched

    run_measure(
        viewer,
        widget,
        labels_data=labels,
        label_dims=["Y", "X"],
        intensity_data=intensity,
        intensity_dims=["Z", "Y", "C"],
    )

    # QMessageBox was called
    assert "title" in called, "Expected QMessageBox to be called"
    assert "Region properties could not be computed" in called["text"]

    # properties remain empty
    assert len(widget.layer.properties) == 0
