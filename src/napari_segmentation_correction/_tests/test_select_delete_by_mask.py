# test_process_action.py

import dask.array as da
import numpy as np
import pytest

from napari_segmentation_correction.process_actions_helpers import process_action
from napari_segmentation_correction.select_delete_widget import (
    delete_labels_by_mask,
    filter_labels_by_mask,
)


@pytest.fixture
def small_seg_np():
    """
    3 frames, shape = (T, X, Y)
    Label 1 in frame 0, label 2 in frame 1, label 3 in frame 2.
    """
    arr = np.zeros((3, 5, 5), dtype=int)
    arr[0, 1:3, 1:3] = 1
    arr[1, 1:3, 1:3] = 2
    arr[2, 1:3, 1:3] = 3
    return arr


@pytest.fixture
def small_seg_dask(small_seg_np):
    return da.from_array(small_seg_np, chunks=(1, 5, 5))


@pytest.fixture
def small_mask():
    """A mask overlapping the center."""
    mask = np.zeros((5, 5), dtype=int)
    mask[1:3, 1:3] = 1
    return mask


@pytest.fixture
def tmp_output(tmp_path):
    """Temp directory used when Dask output is required."""
    return tmp_path


# NUMPY: SINGLE FRAME, NOT IN PLACE
def test_numpy_single_not_inplace(small_seg_np, small_mask):
    out = process_action(
        seg=small_seg_np,
        mask=small_mask,
        action=delete_labels_by_mask,
        seg_index=0,
        mask_index=None,
        in_place=False,
    )

    # Original array unchanged
    assert np.any(out != small_seg_np)
    assert small_seg_np[0, 1, 1] == 1

    # Deleted label 1 in frame 0
    assert np.all(out[0] == 0)
    # Other frames unchanged
    assert np.all(out[1] == small_seg_np[1])
    assert np.all(out[2] == small_seg_np[2])


# NUMPY: SINGLE FRAME, IN PLACE
def test_numpy_single_inplace(small_seg_np, small_mask):
    seg_copy = small_seg_np.copy()
    out = process_action(
        seg=seg_copy,
        mask=small_mask,
        action=filter_labels_by_mask,
        seg_index=1,
        mask_index=None,
        in_place=True,
    )

    # in_place returns the same array object
    assert out is seg_copy

    # frame 1: keep label 2 only → unchanged because mask overlaps fully
    assert np.any(out[1] != 0)
    # frame 0 unchanged
    assert np.any(out[0] != 0)
    # frame 2 unchanged
    assert np.any(out[2] != 0)


# NUMPY: MANY FRAMES, NOT IN PLACE
def test_numpy_many_not_in_place(small_seg_np, small_mask):
    out = process_action(
        seg=small_seg_np,
        mask=small_mask,
        action=delete_labels_by_mask,
        seg_index=range(3),
        mask_index=None,
        in_place=False,
    )
    # All labels deleted
    assert np.all(out == 0)
    # Original unchanged
    assert np.any(small_seg_np != 0)


# NUMPY: MANY FRAMES W/ MASK INDEX
def test_numpy_many_with_mask_index(small_seg_np, small_mask):
    masks = np.stack([small_mask, small_mask, small_mask])
    out = process_action(
        seg=small_seg_np.copy(),
        mask=masks,
        action=filter_labels_by_mask,
        seg_index=[0, 1, 2],
        mask_index=[0, 1, 2],
        in_place=False,
    )
    # Each frame keeps its labels
    assert np.any(out[0] != 0)
    assert np.any(out[1] != 0)
    assert np.any(out[2] != 0)

    out = process_action(
        seg=small_seg_np.copy(),
        mask=masks,
        action=delete_labels_by_mask,
        seg_index=[0, 1, 2],
        mask_index=[0, 1, 2],
        in_place=False,
    )
    # Each frame has its labels deleted
    assert np.all(out[0] == 0)
    assert np.all(out[1] == 0)
    assert np.all(out[2] == 0)


# DASK: SINGLE FRAME
def test_dask_single(tmp_output, small_seg_dask, small_mask, monkeypatch):
    # monkeypatch QFileDialog to auto-return tmp_output
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *a, **k: str(tmp_output),
    )

    out = process_action(
        seg=small_seg_dask,
        mask=small_mask,
        action=delete_labels_by_mask,
        seg_index=1,
        mask_index=None,
        basename="test",
        in_place=False,
    )

    # out is new dask array
    assert isinstance(out, da.Array)
    # frame 1 cleared
    assert np.all(out[1].compute() == 0)
    # frame 0/2 unchanged
    assert np.any(out[0].compute() != 0)
    assert np.any(out[2].compute() != 0)


# DASK: ALL FRAMES
def test_dask_many(tmp_output, small_seg_dask, small_mask, monkeypatch):
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *a, **k: str(tmp_output),
    )

    out = process_action(
        seg=small_seg_dask,
        mask=small_mask,
        action=delete_labels_by_mask,
        seg_index=range(3),
        mask_index=None,
        basename="test",
        in_place=False,
    )

    assert isinstance(out, da.Array)
    assert out.shape == (3, 5, 5)
    assert np.all(out.compute() == 0)


# DASK: MANY FRAMES WITH MASK INDEX
def test_dask_many_with_mask_index(tmp_output, small_seg_dask, small_mask, monkeypatch):
    monkeypatch.setattr(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *a, **k: str(tmp_output),
    )

    # stack mask
    masks = da.from_array(
        np.stack([small_mask, small_mask, small_mask]), chunks=(1, 5, 5)
    )

    out = process_action(
        seg=small_seg_dask,
        mask=masks,
        action=filter_labels_by_mask,
        seg_index=[0, 1, 2],
        mask_index=[0, 1, 2],
        basename="test",
        in_place=False,
    )

    assert isinstance(out, da.Array)
    assert np.any(out[0].compute() != 0)
    assert np.any(out[1].compute() != 0)
    assert np.any(out[2].compute() != 0)

    out = process_action(
        seg=small_seg_dask,
        mask=masks,
        action=delete_labels_by_mask,
        seg_index=[0, 1, 2],
        mask_index=[0, 1, 2],
        basename="test",
        in_place=False,
    )

    assert isinstance(out, da.Array)
    assert np.all(out[0].compute() == 0)
    assert np.all(out[1].compute() == 0)
    assert np.all(out[2].compute() == 0)
