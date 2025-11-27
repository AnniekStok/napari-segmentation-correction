from unittest.mock import patch

import numpy as np
import pytest

from napari_segmentation_correction.helpers.process_actions_helpers import (
    process_action,
    process_action_seg,
)
from napari_segmentation_correction.tool_widgets.connected_components import (
    connected_component_labeling,
    keep_largest_cluster,
    keep_largest_fragment_per_label,
)
from napari_segmentation_correction.tool_widgets.erosion_dilation_widget import (
    erode_labels,
    expand_labels_skimage,
)
from napari_segmentation_correction.tool_widgets.image_calculator import (
    add_images,
    divide_images,
    logical_and,
    logical_or,
    multiply_images,
    subtract_images,
)
from napari_segmentation_correction.tool_widgets.label_boundaries import (
    compute_boundaries,
)
from napari_segmentation_correction.tool_widgets.smoothing_widget import median_filter
from napari_segmentation_correction.tool_widgets.threshold_widget import threshold


def test_threshold(img_2d, img_3d, img_4d, tmp_path):
    """Test smoothing with median filter"""

    action = threshold
    min_val = 2
    max_val = 3
    result = process_action_seg(
        img_2d, action, basename="2d_img", min_val=min_val, max_val=max_val
    )

    expected = action(img_2d, min_val, max_val)
    np.testing.assert_array_equal(result, expected)
    assert np.issubdtype(result.dtype, np.bool_)
    assert result[6, 12] == 0
    assert result[6, 5] == 1

    img = img_3d()
    result = process_action_seg(
        img, action, basename="3d_img", min_val=min_val, max_val=max_val
    )

    expected = action(img, min_val, max_val)
    np.testing.assert_array_equal(result, expected)

    img = img_4d()
    result = process_action_seg(
        img, action, basename="4d_img", min_val=min_val, max_val=max_val
    )

    expected = action(img[0], min_val, max_val)
    np.testing.assert_array_equal(result[0], expected)

    img = img_4d(dask=True)
    with patch(
        "napari_segmentation_correction.helpers.process_actions_helpers.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)

        result = process_action_seg(
            img, action, basename="4d_img_dask", min_val=min_val, max_val=max_val
        )

    expected = action(img[0].compute(), min_val, max_val)
    np.testing.assert_array_equal(result[0], expected)


def test_median_filter(img_2d, img_3d, img_4d, tmp_path):
    """Test smoothing with median filter"""

    action = median_filter
    size = 3
    result = process_action_seg(img_2d, action, basename="2d_img", size=size)

    expected = action(img_2d, size)
    np.testing.assert_array_equal(result, expected)
    assert result[6, 12] == 0

    img = img_3d()
    result = process_action_seg(img, action, basename="3d_img", size=size)

    expected = action(img, size)
    np.testing.assert_array_equal(result, expected)

    img = img_4d()
    result = process_action_seg(img, action, basename="4d_img", size=size)

    expected = action(img[0], size)
    np.testing.assert_array_equal(result[0], expected)

    img = img_4d(dask=True)
    with patch(
        "napari_segmentation_correction.helpers.process_actions_helpers.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)

        result = process_action_seg(img, action, basename="4d_img_dask", size=size)

    expected = action(img[0].compute(), size=size)
    np.testing.assert_array_equal(result[0], expected)


@pytest.mark.parametrize("action", [erode_labels, expand_labels_skimage])
def test_erode_dilate(action, img_2d, img_3d, img_4d, tmp_path):
    """Test erosion/dilation"""

    result = process_action_seg(img_2d, action, basename="2d_img", diam=3, iterations=1)

    expected = action(img_2d, diam=3, iterations=1)
    np.testing.assert_array_equal(result, expected)

    if action == erode_labels:
        assert result[5, 3] == 0
    elif action == expand_labels_skimage:
        assert result[2, 1] == 2

    img = img_3d()
    result = process_action_seg(img, action, basename="3d_img", diam=3, iterations=1)

    expected = action(img, diam=3, iterations=1)
    np.testing.assert_array_equal(result, expected)

    img = img_4d()
    result = process_action_seg(img, action, basename="4d_img", diam=3, iterations=1)

    expected = action(img[0], diam=3, iterations=1)
    np.testing.assert_array_equal(result[0], expected)

    img = img_4d(dask=True)
    with patch(
        "napari_segmentation_correction.helpers.process_actions_helpers.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)

        result = process_action_seg(
            img, action, basename="4d_img_dask", diam=3, iterations=1
        )

    expected = action(img[0].compute(), diam=3, iterations=1)
    np.testing.assert_array_equal(result[0], expected)


@pytest.mark.parametrize(
    "action",
    [
        compute_boundaries,
        keep_largest_cluster,
        keep_largest_fragment_per_label,
        connected_component_labeling,
    ],
)
def test_conn_comp_boundaries(action, img_2d, img_3d, img_4d, tmp_path):
    """Test connected component function and boundary computation"""
    result = process_action_seg(
        img_2d,
        action,
        basename="2d_img",
    )

    expected = action(img_2d)
    np.testing.assert_array_equal(result, expected)

    if action == compute_boundaries:
        assert result[6, 5] == 0
    elif action == keep_largest_cluster:
        assert result[6, 14] == 0
        assert result[6, 5] == 2
    elif action == keep_largest_fragment_per_label:
        assert result[9, 16] == 0
    elif action == connected_component_labeling:
        assert result[9, 16] == 3

    img = img_3d()
    result = process_action_seg(
        img,
        action,
        basename="3d_img",
    )

    expected = action(img)
    np.testing.assert_array_equal(result, expected)

    img = img_4d()
    result = process_action_seg(
        img,
        action,
        basename="4d_img",
    )

    expected = action(img[0])
    np.testing.assert_array_equal(result[0], expected)

    img = img_4d(dask=True)
    with patch(
        "napari_segmentation_correction.helpers.process_actions_helpers.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)

        result = process_action_seg(
            img,
            action,
            basename="4d_img_dask",
        )

    expected = action(img[0].compute())
    np.testing.assert_array_equal(result[0], expected)


@pytest.mark.parametrize(
    "action",
    [
        add_images,
        subtract_images,
        multiply_images,
        divide_images,
        logical_and,
        logical_or,
    ],
)
@pytest.mark.parametrize("adjust_dtype", [True, False])
def test_image_calculator(action, adjust_dtype, img_2d, img_3d, img_4d, tmp_path):
    """Test image calculator functions"""

    img1 = img_2d
    img2 = np.rot90(img_2d)
    result = process_action(
        img1=img1,
        img2=img2,
        action=action,
        basename="img_2d",
        adjust_dtype=adjust_dtype,
    )
    expected = action(img1=img1, img2=img2, adjust_dtype=adjust_dtype)
    np.testing.assert_array_equal(result, expected)

    if action == add_images and adjust_dtype:
        assert result[14, 17] == 258
        assert result.dtype == np.uint16
    elif action == add_images and not adjust_dtype:
        assert result[14, 17] == 255
        assert result.dtype == img1.dtype
    elif action == subtract_images and adjust_dtype:
        assert result[6, 16] == -3
        assert result.dtype == np.int16
    elif action == subtract_images and not adjust_dtype:
        assert result[6, 16] == 0
        assert result.dtype == img1.dtype
    elif action == multiply_images and adjust_dtype:
        assert result[14, 17] == 765
        assert result.dtype == np.uint16
    elif action == multiply_images and not adjust_dtype:
        assert result[14, 17] == 255
        assert result.dtype == img1.dtype
    elif action == divide_images and adjust_dtype:
        assert result[6, 14] == 1 / 3
        assert result.dtype == np.float64
    elif action == divide_images and not adjust_dtype:
        assert result[6, 14] == 0
        assert result.dtype == img1.dtype
    elif action == logical_and:
        assert result[6, 14] == 1
        assert result[6, 15] == 0
    elif action == logical_or:
        assert result[6, 14] == 1
        assert result[6, 15] == 1

    img1 = img_3d()
    img2 = np.rot90(img1)
    result = process_action(img1=img1, img2=img2, action=action, basename="img_3d")
    expected = action(img1, img2)
    np.testing.assert_array_equal(result, expected)

    img1 = img_4d()
    result = process_action(img1=img1, img2=img1, action=action, basename="img_4d")
    expected = action(img1, img1)
    np.testing.assert_array_equal(result, expected)

    img1 = img_3d(dask=True)
    img2 = np.rot90(img1)
    indices = range(img1.shape[0])
    with patch(
        "napari_segmentation_correction.helpers.process_actions_helpers.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)
        result = process_action(
            img1=img1,
            img2=img2,
            action=action,
            basename="img_3d",
            img1_index=indices,
            img2_index=indices,
        )
    expected = action(img1[0], img2[0])
    np.testing.assert_array_equal(result[0], expected)

    img1 = img_4d(dask=True)
    indices = range(img1.shape[0])
    with patch(
        "napari_segmentation_correction.helpers.process_actions_helpers.QFileDialog.getExistingDirectory"
    ) as mock_dialog:
        mock_dialog.return_value = str(tmp_path)
        result = process_action(
            img1=img1,
            img2=img1,
            action=action,
            basename="img_3d",
            img1_index=indices,
            img2_index=indices,
        )
    expected = action(img1[0], img1[0])
    np.testing.assert_array_equal(result[0], expected)
