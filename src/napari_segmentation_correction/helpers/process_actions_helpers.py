import copy
import os
import re

import dask.array as da
import numpy as np
import tifffile
from dask import delayed
from napari_builtins.io._read import (
    magic_imread,
)
from qtpy.QtWidgets import (
    QFileDialog,
)


def remove_invalid_chars(name: str) -> str:
    "Remove invalid characters"
    return re.sub(r'[\\/:*?"<>|\[\]]+', "", name)


def apply_action(
    img1: np.ndarray, img2: np.ndarray | None, action: callable, **action_kwargs
) -> np.ndarray:
    """
    Apply an action to a single 2D/3D numpy array.
    'action' is a function that takes one or two inputs and returns the processed image.
    """
    return (
        action(img1, img2, **action_kwargs)
        if img2 is not None
        else action(img1, **action_kwargs)
    )


def merge_modified_slices(
    original: da.Array, modified: dict[int, np.ndarray]
) -> da.Array:
    """
    Create a new dask array identical to `original` but with some slices
    replaced by modified numpy arrays.

    Args:
        original (da.Array):
            The input 3D or 4D array: (T, Z, Y, X) or (T, Y, X)
        modified (dict[int, np.ndarray]):
            Keys are indices along axis=0 that were modified.
            Values are the modified numpy slices.

    Returns
        da.Array: A new lazily loaded array with patched slices.
    """

    slices = []

    for i in range(original.shape[0]):
        if i in modified:
            # Modified slice: wrap as delayed object
            arr = modified[i]
            delayed_slice = delayed(lambda x: x)(arr)  # keep lazy
            d = da.from_delayed(delayed_slice, shape=arr.shape, dtype=arr.dtype)

        else:
            # Unmodified: use original lazy dask slice
            d = original[i, ...]

        slices.append(d)

    return da.stack(slices, axis=0)


def process_action(
    img1: np.ndarray | da.core.Array,
    img2: np.ndarray | da.core.Array,
    action: callable,
    img1_index: int | list[int] | None = None,
    img2_index: int | list[int] | None = None,
    basename: str | None = None,
    in_place: bool = False,
    **action_kwargs,
) -> da.core.Array | np.ndarray:
    """
    Process a dask array segmentation with given img2 and action.
    If img1_index and img2_index are both provided, they should be iterables of the same length.
    If only img1_index is provided, img2 is assumed to be 2D/3D and applied to each img1 slice.
    Returns a dask array with processed data.
    """

    if isinstance(img1, np.ndarray) and not in_place:
        img1 = copy.deepcopy(img1)

    if isinstance(img1, da.core.Array) and isinstance(img1_index, (list, range)):
        outputdir = QFileDialog.getExistingDirectory(caption="Select Output Folder")
        if not outputdir:
            return

        outputdir = os.path.join(
            outputdir, remove_invalid_chars(basename + "_" + str(action.__name__))
        )

        while os.path.exists(outputdir):
            outputdir = outputdir + "_1"

        os.mkdir(outputdir)

    # process single frame
    if isinstance(img1_index, int):
        if isinstance(img1, da.core.Array):
            img1_frame = img1[img1_index].compute()
            modified = {}
        else:
            img1_frame = img1[img1_index]

        if isinstance(img2_index, int):
            if isinstance(img2, da.core.Array):
                img2_frame = img2[img2_index].compute()
            else:
                img2_frame = img2[img2_index]
        else:
            img2_frame = img2

        processed = apply_action(img1_frame, img2_frame, action, **action_kwargs)

        if isinstance(img1, da.core.Array):
            modified[img1_index] = processed
            # update dask array
            return merge_modified_slices(img1, modified)
        else:
            img1[img1_index] = processed
            return img1

    # process all frames
    elif isinstance(img1_index, (list, range)):
        if isinstance(img1, da.core.Array):
            # dask array
            if img2_index is not None and isinstance(img2_index, (list, range)):
                # both img1 and img2 are indexed
                for i, j in zip(img1_index, img2_index, strict=True):
                    img1_frame = img1[i].compute()
                    if isinstance(img2, da.core.Array):
                        img2_frame = img2[j].compute()
                    else:
                        img2_frame = img2[j]
                    processed = apply_action(
                        img1_frame, img2_frame, action, **action_kwargs
                    )

                    fname = remove_invalid_chars(f"{basename}{str(i).zfill(4)}.tif")
                    path = os.path.join(outputdir, fname)

                    tifffile.imwrite(path, processed)

                return magic_imread(outputdir, use_dask=True)

            else:
                # only img1 is indexed, img2 is 2D/3D
                for i in img1_index:
                    img1_frame = img1[i].compute()
                    processed = apply_action(img1_frame, img2, action, **action_kwargs)

                    fname = remove_invalid_chars(f"{basename}{str(i).zfill(4)}.tif")
                    path = os.path.join(outputdir, fname)

                    tifffile.imwrite(path, processed)
                return magic_imread(outputdir, use_dask=True)

        else:
            # numpy array
            if img2_index is not None and isinstance(img2_index, (list, range)):
                # both img1 and img2 are indexed
                for i, j in zip(img1_index, img2_index, strict=True):
                    img1_frame = img1[i]
                    img2_frame = img2[j]
                    processed = apply_action(
                        img1_frame, img2_frame, action, **action_kwargs
                    )
                    img1[i] = processed
                return img1
            else:
                # only img1 is indexed, img2 is 2D/3D
                for i in img1_index:
                    img1_frame = img1[i]
                    processed = apply_action(img1_frame, img2, action, **action_kwargs)
                    img1[i] = processed
                return img1

    elif img1_index is None and img2_index is None:
        # process entire stack
        processed = apply_action(img1, img2, action, **action_kwargs)
        return processed


def process_action_seg(
    seg: np.ndarray | da.core.Array,
    action: callable,
    basename: str | None = None,
    in_place: bool = False,
    **action_kwargs,
) -> da.core.Array | np.ndarray:
    """
    Process a dask array segmentation with given img2 and action.
    If seg_index and img2_index are both provided, they should be iterables of the same length.
    If only seg_index is provided, img2 is assumed to be 2D/3D and applied to each seg slice.
    Returns a dask array with processed data.
    """

    if isinstance(seg, np.ndarray) and not in_place:
        seg = copy.deepcopy(seg)

    if isinstance(seg, da.core.Array):
        outputdir = QFileDialog.getExistingDirectory(caption="Select Output Folder")
        if not outputdir:
            return

        outputdir = os.path.join(
            outputdir, remove_invalid_chars(basename + "_" + str(action.__name__))
        )

        while os.path.exists(outputdir):
            outputdir = outputdir + "_1"

        os.mkdir(outputdir)

    # process all frames
    if isinstance(seg, da.core.Array):
        # dask array, loop over frames
        for i in range(seg.shape[0]):
            seg_frame = seg[i].compute()
            processed = apply_action(seg_frame, None, action, **action_kwargs)
            fname = remove_invalid_chars(f"{basename}{str(i).zfill(4)}.tif")
            path = os.path.join(outputdir, fname)
            tifffile.imwrite(path, processed)
        return magic_imread(outputdir, use_dask=True)
    else:
        # numpy array
        if seg.ndim == 4:
            # loop over all frames if shape == 4
            for i in range(seg.shape[0]):
                seg_frame = seg[i]
                processed = apply_action(seg_frame, None, action, **action_kwargs)
                seg[i] = processed
            return seg
        else:
            return apply_action(seg, None, action, **action_kwargs)
