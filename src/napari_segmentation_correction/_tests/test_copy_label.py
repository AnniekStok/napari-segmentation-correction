import numpy as np
import pytest
from napari.layers import Labels

from napari_segmentation_correction.copy_label_widget import CopyLabelWidget


@pytest.fixture
def make_event():
    def _make_event(position, modifiers=("Shift",), dims_displayed=(0, 1)):
        class DummyEvent:
            def __init__(self):
                self.type = "mouse_press"
                self.position = position
                self.modifiers = modifiers
                self.dims_displayed = dims_displayed
                self.view_direction = None

            # mimic napari Event

        return DummyEvent()

    return _make_event


@pytest.fixture
def make_layers():
    def _make_layers(shape):
        src = Labels(np.zeros(shape, dtype=np.uint16))
        tgt = Labels(np.zeros(shape, dtype=np.uint16))
        return src, tgt

    return _make_layers


def test_copy_2d(make_event, make_napari_viewer):
    viewer = make_napari_viewer()
    src = viewer.add_labels(np.zeros((64, 64), dtype=np.uint16))
    tgt = viewer.add_labels(np.zeros((64, 64), dtype=np.uint16))
    src.data[30:40, 30:40] = 1

    widget = CopyLabelWidget(viewer)
    widget.source_layer = src
    widget.target_layer = tgt
    widget.dims_widget.slice.setChecked(True)

    event = make_event([32, 32])
    widget.copy_label(event)

    expected = np.zeros((64, 64), dtype=np.uint8)
    expected[30:40, 30:40] = 1
    np.testing.assert_array_equal(tgt.data, expected)


def test_copy_3d(make_event, make_napari_viewer):
    viewer = make_napari_viewer()
    src = viewer.add_labels(np.zeros((10, 64, 64), dtype=np.uint16))
    tgt = viewer.add_labels(np.zeros((10, 64, 64), dtype=np.uint16))
    src.data[5:8, 20:30, 20:30] = 10

    widget = CopyLabelWidget(viewer)
    widget.source_layer = src
    widget.target_layer = tgt
    widget.dims_widget.volume.setChecked(True)

    viewer.dims.current_step = (5, 0, 0)

    event = make_event([5, 25, 25], dims_displayed=[1, 2])
    widget.copy_label(event)

    # check copying volume
    expected = np.zeros((10, 64, 64), dtype=np.uint8)
    expected[5:8, 20:30, 20:30] = 1
    np.testing.assert_array_equal(tgt.data, expected)

    # check preserving label value
    widget.preserve_label_value.setChecked(True)
    widget.copy_label(event)
    expected[5:8, 20:30, 20:30] = 10
    np.testing.assert_array_equal(tgt.data, expected)

    # check replacing a slice only
    widget.dims_widget.slice.setChecked(True)
    widget.preserve_label_value.setChecked(False)
    widget.copy_label(event)
    expected[5, 20:30, 20:30] = 11
    np.testing.assert_array_equal(tgt.data, expected)


def test_copy_4d(make_event, make_napari_viewer):
    viewer = make_napari_viewer()
    src = viewer.add_labels(np.zeros((3, 10, 64, 64), dtype=np.uint16))
    tgt = viewer.add_labels(np.zeros((3, 10, 64, 64), dtype=np.uint16))
    src.data[1:2, 5:8, 20:30, 20:30] = 10

    widget = CopyLabelWidget(viewer)
    widget.source_layer = src
    widget.target_layer = tgt
    widget.dims_widget.series.setChecked(True)

    viewer.dims.current_step = (1, 5, 0, 0)

    event = make_event([1, 5, 25, 25], dims_displayed=[2, 3])
    widget.copy_label(event)
    expected = np.zeros((3, 10, 64, 64), dtype=np.uint8)
    expected[1:2, 5:8, 20:30, 20:30] = 1
    np.testing.assert_array_equal(tgt.data, expected)

    widget.preserve_label_value.setChecked(True)
    widget.copy_label(event)
    expected[1:2, 5:8, 20:30, 20:30] = 10
    np.testing.assert_array_equal(tgt.data, expected)


def test_copy_4d_slice_to_2d(make_event, make_napari_viewer):
    viewer = make_napari_viewer()
    src = viewer.add_labels(np.zeros((3, 10, 64, 64), dtype=np.uint16))
    tgt = Labels(np.zeros((64, 64), dtype=np.uint16))
    src.data[1:2, 5:8, 20:30, 20:30] = 10

    widget = CopyLabelWidget(viewer)
    widget.source_layer = src
    widget.target_layer = tgt
    widget.dims_widget.slice.setChecked(True)

    viewer.dims.current_step = (1, 5, 0, 0)

    event = make_event([1, 5, 25, 25], dims_displayed=[2, 3])
    widget.copy_label(event)
    expected = np.zeros((64, 64), dtype=np.uint8)
    expected[20:30, 20:30] = 1
    np.testing.assert_array_equal(tgt.data, expected)

    widget.preserve_label_value.setChecked(True)
    widget.copy_label(event)
    expected[20:30, 20:30] = 10
    np.testing.assert_array_equal(tgt.data, expected)


def test_copy_2d_slice_to_4d(make_event, make_napari_viewer):
    viewer = make_napari_viewer()
    src = Labels(np.zeros((64, 64), dtype=np.uint16))
    tgt = viewer.add_labels(np.zeros((3, 10, 64, 64), dtype=np.uint16))
    src.data[20:30, 20:30] = 10

    widget = CopyLabelWidget(viewer)
    widget.source_layer = src
    widget.target_layer = tgt
    widget.dims_widget.slice.setChecked(True)

    viewer.dims.current_step = (2, 5, 0, 0)

    event = make_event([2, 5, 25, 25], dims_displayed=[2, 3])
    widget.copy_label(event)
    expected = np.zeros((3, 10, 64, 64), dtype=np.uint8)
    expected[2, 5, 20:30, 20:30] = 1
    np.testing.assert_array_equal(tgt.data, expected)

    widget.preserve_label_value.setChecked(True)
    widget.copy_label(event)
    expected[2, 5, 20:30, 20:30] = 10
    np.testing.assert_array_equal(tgt.data, expected)
