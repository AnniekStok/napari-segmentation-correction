from pathlib import Path

import pytest
import tifffile
from napari_builtins.io._read import (
    magic_imread,
)


@pytest.fixture
def img_2d(request):
    test_dir = Path(request.node.fspath).parent

    return tifffile.imread(test_dir / "test2d.tif")


@pytest.fixture
def img_3d(request):
    test_dir = Path(request.node.fspath).parent

    def _factory(dask=False):
        return magic_imread(test_dir / "test3d.tif", use_dask=dask)

    return _factory


@pytest.fixture
def img_4d(request):
    test_dir = Path(request.node.fspath).parent

    def _factory(dask=False):
        return magic_imread(test_dir / "test4d.tif", use_dask=dask)

    return _factory
