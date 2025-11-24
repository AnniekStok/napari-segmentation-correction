import copy
import os

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


def apply_action(
    image: np.ndarray, mask: np.ndarray | None, action: callable, **action_kwargs
) -> np.ndarray:
    """
    Apply an action to a single 2D/3D numpy array.
    'action' is a function that takes one or two inputs and returns the processed image.
    """
    return (
        action(image, mask, **action_kwargs)
        if mask is not None
        else action(image, **action_kwargs)
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
    seg: np.ndarray | da.core.Array,
    mask: np.ndarray | da.core.Array,
    action: callable,
    seg_index: int | list[int],
    mask_index: int | list[int],
    basename: str | None = None,
    in_place: bool = False,
    **action_kwargs,
) -> da.core.Array | np.ndarray:
    """
    Process a dask array segmentation with given mask and action.
    If seg_index and mask_index are both provided, they should be iterables of the same length.
    If only seg_index is provided, mask is assumed to be 2D/3D and applied to each seg slice.
    Returns a dask array with processed data.
    """

    if isinstance(seg, np.ndarray) and not in_place:
        seg = copy.deepcopy(seg)

    if isinstance(seg, da.core.Array) and isinstance(seg_index, (list, range)):
        outputdir = QFileDialog.getExistingDirectory(caption="Select Output Folder")
        if not outputdir:
            return

        outputdir = os.path.join(
            outputdir,
            (basename + "_filtered_labels"),
        )

        while os.path.exists(outputdir):
            outputdir = outputdir + "_1"
        os.mkdir(outputdir)

    # process single frame
    if isinstance(seg_index, int):
        if isinstance(seg, da.core.Array):
            seg_frame = seg[seg_index].compute()
            modified = {}
        else:
            seg_frame = seg[seg_index]

        if isinstance(mask_index, int):
            if isinstance(mask, da.core.Array):
                mask_frame = mask[mask_index].compute()
            else:
                mask_frame = mask[mask_index]
        else:
            mask_frame = mask

        processed = apply_action(seg_frame, mask_frame, action, **action_kwargs)

        if isinstance(seg, da.core.Array):
            modified[seg_index] = processed
            # update dask array
            return merge_modified_slices(seg, modified)
        else:
            seg[seg_index] = processed
            return seg

    # process all frames
    elif isinstance(seg_index, (list, range)):
        if isinstance(seg, da.core.Array):
            # dask array
            if mask_index is not None and isinstance(mask_index, (list, range)):
                # both seg and mask are indexed
                for i, j in zip(seg_index, mask_index, strict=True):
                    seg_frame = seg[i].compute()
                    if isinstance(mask, da.core.Array):
                        mask_frame = mask[j].compute()
                    else:
                        mask_frame = mask[j]
                    processed = apply_action(
                        seg_frame, mask_frame, action, **action_kwargs
                    )

                    fname = f"{basename}{str(i).zfill(4)}.tif"
                    path = os.path.join(outputdir, fname)

                    tifffile.imwrite(path, processed)

                return magic_imread(outputdir, use_dask=True)

            else:
                # only seg is indexed, mask is 2D/3D
                for i in seg_index:
                    seg_frame = seg[i].compute()
                    processed = apply_action(seg_frame, mask, action, **action_kwargs)

                    fname = f"{basename}{str(i).zfill(4)}.tif"
                    path = os.path.join(outputdir, fname)

                    tifffile.imwrite(path, processed)
                return magic_imread(outputdir, use_dask=True)

        else:
            # numpy array
            if mask_index is not None and isinstance(mask_index, (list, range)):
                # both seg and mask are indexed
                for i, j in zip(seg_index, mask_index, strict=True):
                    seg_frame = seg[i]
                    mask_frame = mask[j]
                    processed = apply_action(
                        seg_frame, mask_frame, action, **action_kwargs
                    )
                    seg[i] = processed
                return seg
            else:
                # only seg is indexed, mask is 2D/3D
                for i in seg_index:
                    seg_frame = seg[i]
                    processed = apply_action(seg_frame, mask, action, **action_kwargs)
                    seg[i] = processed
                return seg


def process_action_seg(
    seg: np.ndarray | da.core.Array,
    action: callable,
    basename: str | None = None,
    in_place: bool = False,
    **action_kwargs,
) -> da.core.Array | np.ndarray:
    """
    Process a dask array segmentation with given mask and action.
    If seg_index and mask_index are both provided, they should be iterables of the same length.
    If only seg_index is provided, mask is assumed to be 2D/3D and applied to each seg slice.
    Returns a dask array with processed data.
    """

    if isinstance(seg, np.ndarray) and not in_place:
        seg = copy.deepcopy(seg)

    if isinstance(seg, da.core.Array):
        outputdir = QFileDialog.getExistingDirectory(caption="Select Output Folder")
        if not outputdir:
            return

        outputdir = os.path.join(outputdir, (basename + "_" + str(action.__name__)))

        while os.path.exists(outputdir):
            outputdir = outputdir + "_1"
        os.mkdir(outputdir)

    # process all frames
    if isinstance(seg, da.core.Array):
        # dask array, loop over frames
        for i in range(seg.shape[0]):
            seg_frame = seg[i].compute()
            processed = apply_action(seg_frame, None, action, **action_kwargs)
            fname = f"{basename}{str(i).zfill(4)}.tif"
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
