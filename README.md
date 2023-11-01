# napari-segmentation-correction

[![License BSD-3](https://img.shields.io/pypi/l/napari-segmentation-correction.svg?color=green)](https://github.com/AnniekStok/napari-segmentation-correction/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-segmentation-correction.svg?color=green)](https://pypi.org/project/napari-segmentation-correction)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-segmentation-correction.svg?color=green)](https://python.org)
[![tests](https://github.com/AnniekStok/napari-segmentation-correction/workflows/tests/badge.svg)](https://github.com/AnniekStok/napari-segmentation-correction/actions)
[![codecov](https://codecov.io/gh/AnniekStok/napari-segmentation-correction/branch/main/graph/badge.svg)](https://codecov.io/gh/AnniekStok/napari-segmentation-correction)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-segmentation-correction)](https://napari-hub.org/plugins/napari-segmentation-correction)

A basis for a napari plugin for manually correcting cell segmentation in 3D (z, y, x) or 4D (t, z, y, x) (virtual) arrays. 

The plugin heavily relies on several very useful open source packages and other napari plugins:
- orthogonal view widget: [napari multiple viewer widget](https://github.com/napari/napari/blob/e490e5535438ab338a23b17905a1952f15a6d27a/examples/multiple_viewer_widget.py)
- table widget: [napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)
----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-segmentation-correction` via [pip]:

To install latest development version :

    pip install git+https://github.com/AnniekStok/napari-segmentation-correction.git

## Usage
This plugin aims to help you correct segmentation results. It can work with 3D arrays, 4D arrays, or 4D virtual arrays. There are several functionalities: 
- explore label properties in a table widget
- filter labels by size
- select/delete labels with point layer selection
- copy labels from one array to another with point layer selection
- erode/dilate labels
- smooth labels

![](instructions/napari-ndlabelcorrection_filter_by_size.gif)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-segmentation-correction" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/AnniekStok/napari-segmentation-correction/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
