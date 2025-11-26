from unittest.mock import patch

import dask.array as da
import numpy as np
from dask import delayed

from napari_segmentation_correction.process_actions_helpers import (
    process_action,
    process_action_seg,
)


def dummy_action(img, **kwargs):
    """Simple action that returns a modified copy"""
    return img * 2


def dummy_action_with_mask(img, mask, **kwargs):
    """Action that applies a mask"""
    return img * mask


def dummy_action_with_kwargs(img, value, **kwargs):
    """Action that multiplies by value"""
    return img * value


class TestProcessActionSeg2D:
    """Test process_action_seg with 2D numpy arrays"""

    def test_2d_numpy_simple(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = process_action_seg(img, dummy_action, in_place=False)
        expected = np.array([[2, 4], [6, 8]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)

    def test_2d_numpy_with_kwargs(self):
        img = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        result = process_action_seg(
            img, dummy_action_with_kwargs, value=3, in_place=False
        )
        expected = np.array([[3, 3, 3], [3, 3, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_2d_numpy_in_place(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = process_action_seg(img, dummy_action, in_place=True)
        np.testing.assert_array_equal(result, np.array([[2, 4], [6, 8]]))


class TestProcessActionSeg4D:
    """Test process_action_seg with 4D numpy and dask arrays"""

    def test_4d_numpy_simple(self):
        img = np.ones((2, 3, 4, 4), dtype=np.uint16)
        result = process_action_seg(img, dummy_action, in_place=False)
        expected = np.ones((2, 3, 4, 4), dtype=np.uint16) * 2
        np.testing.assert_array_equal(result, expected)

    def test_4d_numpy_with_kwargs(self):
        img = np.ones((2, 3, 5, 5), dtype=np.uint8)
        result = process_action_seg(
            img, dummy_action_with_kwargs, value=3, iterations=1, in_place=False
        )
        expected = np.full((2, 3, 5, 5), 3, dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_4d_dask_simple(self, tmp_path):
        img_np = np.ones((2, 3, 4, 4), dtype=np.uint16)
        img_dask = da.from_delayed(
            delayed(img_np), shape=img_np.shape, dtype=img_np.dtype
        )
        with patch(
            "napari_segmentation_correction.process_actions_helpers.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = str(tmp_path)
            result = process_action_seg(
                img_dask, dummy_action, basename="test_4d", in_place=False
            )
        assert isinstance(result, da.Array)
        np.testing.assert_array_equal(result.compute(), img_np * 2)

    def test_4d_dask_with_kwargs(self, tmp_path):
        img_np = np.ones((2, 3, 5, 5), dtype=np.uint8)
        img_dask = da.from_delayed(
            delayed(img_np), shape=img_np.shape, dtype=img_np.dtype
        )

        with patch(
            "napari_segmentation_correction.process_actions_helpers.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = str(tmp_path)
            result = process_action_seg(
                img_dask,
                dummy_action_with_kwargs,
                basename="test_4d_kwargs",
                value=3,
                in_place=False,
            )
        assert isinstance(result, da.Array)

        expected = np.full((2, 3, 5, 5), 3, dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


class TestProcessAction2D:
    """Test process_action with 2D numpy arrays"""

    def test_2d_numpy_no_indices(self):
        seg = np.ones((4, 4), dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(seg, mask, dummy_action_with_mask, in_place=False)
        np.testing.assert_array_equal(result, seg * mask)

    def test_3d_dask_multiple_frames(self, tmp_path):
        """Test with 3D dask array and multiple frames - mock file dialog"""
        seg_np = np.ones((3, 4, 4), dtype=np.uint16)
        seg_dask = da.from_delayed(
            delayed(seg_np), shape=seg_np.shape, dtype=seg_np.dtype
        )
        mask = np.ones((4, 4), dtype=np.uint8)

        # Mock QFileDialog to return tmp_path instead of blocking
        with patch(
            "napari_segmentation_correction.process_actions_helpers.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = str(tmp_path)
            result = process_action(
                seg_dask,
                mask,
                dummy_action_with_mask,
                img1_index=[0, 1, 2],
                img2_index=None,
                basename="test_3d_multi",
                in_place=False,
            )
            assert isinstance(result, da.Array)


class TestProcessAction3D:
    """Test process_action with 3D numpy and dask arrays"""

    def test_3d_numpy_single_frame(self):
        seg = np.ones((3, 4, 4), dtype=np.uint16)
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(
            seg,
            mask,
            dummy_action_with_mask,
            img1_index=1,
            img2_index=None,
            in_place=False,
        )
        expected = np.ones((3, 4, 4), dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)

    def test_3d_numpy_multiple_frames(self):
        seg = np.ones((3, 4, 4), dtype=np.uint16)
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(
            seg,
            mask,
            dummy_action_with_mask,
            img1_index=[0, 1, 2],
            img2_index=None,
            in_place=False,
        )
        assert result.shape == seg.shape

    def test_3d_dask_single_frame(self):
        seg_np = np.ones((3, 4, 4), dtype=np.uint16)
        seg_dask = da.from_delayed(
            delayed(seg_np), shape=seg_np.shape, dtype=seg_np.dtype
        )
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(
            seg_dask,
            mask,
            dummy_action_with_mask,
            img1_index=1,
            img2_index=None,
            in_place=False,
        )
        assert isinstance(result, (np.ndarray, da.Array))

    def test_3d_dask_multiple_frames(self, tmp_path):
        """Test with 3D dask array and multiple frames - mock file dialog"""
        seg_np = np.ones((3, 4, 4), dtype=np.uint16)
        seg_dask = da.from_delayed(
            delayed(seg_np), shape=seg_np.shape, dtype=seg_np.dtype
        )
        mask = np.ones((4, 4), dtype=np.uint8)

        # Mock QFileDialog to return tmp_path instead of blocking
        with patch(
            "napari_segmentation_correction.process_actions_helpers.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = str(tmp_path)
            result = process_action(
                seg_dask,
                mask,
                dummy_action_with_mask,
                img1_index=[0, 1, 2],
                img2_index=None,
                basename="test_3d_multi",
                in_place=False,
            )
            assert isinstance(result, da.Array)


class TestProcessAction4D:
    """Test process_action with 4D numpy and dask arrays"""

    def test_4d_numpy_single_frame(self):
        seg = np.ones((2, 3, 4, 4), dtype=np.uint16)
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(
            seg,
            mask,
            dummy_action_with_mask,
            img1_index=1,
            img2_index=None,
            in_place=False,
        )
        assert isinstance(result, np.ndarray)

    def test_4d_numpy_multiple_frames(self):
        seg = np.ones((2, 3, 4, 4), dtype=np.uint16)
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(
            seg,
            mask,
            dummy_action_with_mask,
            img1_index=[0, 1],
            img2_index=None,
            in_place=False,
        )
        assert result.shape == seg.shape

    def test_4d_dask_single_frame(self):
        seg_np = np.ones((2, 3, 4, 4), dtype=np.uint16)
        seg_dask = da.from_delayed(
            delayed(seg_np), shape=seg_np.shape, dtype=seg_np.dtype
        )
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(
            seg_dask,
            mask,
            dummy_action_with_mask,
            img1_index=1,
            img2_index=None,
            in_place=False,
        )
        assert isinstance(result, (np.ndarray, da.Array))

    def test_4d_dask_multiple_frames(self, tmp_path):
        """Test with 4D dask array and multiple frames - mock file dialog"""
        seg_np = np.ones((2, 3, 4, 4), dtype=np.uint16)
        seg_dask = da.from_delayed(
            delayed(seg_np), shape=seg_np.shape, dtype=seg_np.dtype
        )
        mask = np.ones((4, 4), dtype=np.uint8)

        with patch(
            "napari_segmentation_correction.process_actions_helpers.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = str(tmp_path)
            result = process_action(
                seg_dask,
                mask,
                dummy_action_with_mask,
                img1_index=[0, 1],
                img2_index=None,
                basename="test_4d_multi",
                in_place=False,
            )
            assert isinstance(result, da.Array)


class TestProcessActionEdgeCases:
    """Test edge cases and error handling"""

    def test_single_element_list_index(self):
        seg = np.ones((3, 4, 4), dtype=np.uint16)
        mask = np.ones((4, 4), dtype=np.uint8)
        result = process_action(
            seg,
            mask,
            dummy_action_with_mask,
            img1_index=[1],
            img2_index=None,
            in_place=False,
        )
        assert result.shape == seg.shape

    def test_no_copy_when_not_in_place(self):
        seg = np.ones((3, 4, 4), dtype=np.uint16)
        mask = np.ones((4, 4), dtype=np.uint8)
        original_id = id(seg)
        result = process_action(
            seg,
            mask,
            dummy_action_with_mask,
            img1_index=[0, 1, 2],
            img2_index=None,
            in_place=False,
        )
        assert id(result) != original_id
