# napari-segmentation-toolbox

[![License BSD-3](https://img.shields.io/pypi/l/napari-segmentation-toolbox.svg?color=green)](https://github.com/AnniekStok/napari-segmentation-toolbox/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-segmentation-toolbox.svg?color=green)](https://pypi.org/project/napari-segmentation-toolbox)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-segmentation-toolbox.svg?color=green)](https://python.org)
[![tests](https://github.com/AnniekStok/napari-segmentation-toolbox/workflows/tests/badge.svg)](https://github.com/AnniekStok/napari-segmentation-toolbox/actions)
[![codecov](https://codecov.io/gh/AnniekStok/napari-segmentation-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/AnniekStok/napari-segmentation-toolbox)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-segmentation-toolbox)](https://napari-hub.org/plugins/napari-segmentation-toolbox)

Toolbox for viewing, analyzing and correcting (cell) segmentation in 2D, 3D or 4D (t, z, y, x) (virtual) arrays.
----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-segmentation-toolbox` via [pip]:

To install latest development version :

    pip install napari-segmentation-toolbox

## Usage
The aim is to serve as a toolbox that provides easy access to functionalities from ![SciPy](https://pypi.org/project/scipy/) and ![scikit-image](https://pypi.org/project/scikit-image/) that can help to explore and correct segmentation data.

- Orthogonal views for 3D data (also available ![separately](https://napari-hub.org/plugins/napari-orthogonal-views.html)).
- Copy labels from a 2-5 dimensional array with multiple segmentation options to your current 2-5 dimensional label layer.
- Label connected components, keep the largest connected cluster of labels, keep the largest fragment per label.
- Smooth labels using a median filter.
- Compute label boundaries.
- Erode/dilate labels (scipy.ndimage and scikit-image).
- Binarize an image or labels layer by applying an intensity threshold.
- Image calculator for mathematical operations between two images.
- Select/delete labels using a mask.
- Binary mask interpolation in the z or time dimension.
- Explore label properties in a table widget and a Matplotlib plot.
- Filter and color regions by properties.

### Copy labels between different labels layers
![copy_labels](https://github.com/user-attachments/assets/4f6a638d-c6bc-4a61-bdcd-6cc29b6f817e)

<table>
  <tr>
    <td width="50%">
      2D/3D/4D labels can be copied from a source layer to a target layer via <b>SHIFT+CLICK</b>.<br><br>
      The data in the source layer should have the same shape as the target layer, but can optionally have one extra dimension (e.g. stack multiple segmentation solutions as channels).<br><br>
      To copy labels, select a 'source' and a 'target' labels layer in the dropdown. By default, the source layer will be displayed as contours.<br><br>
      Select whether to copy a slice, a volume, or a series across time.<br><br>
      Checking <b>Use source label value</b> keeps the original label values.<br><br>
      Selecting <b>Preserve target labels</b> only allows copying into background (0) regions. Otherwise, SHIFT+CLICK replaces the existing label region.
    </td>
    <td width="50%">
      <img alt="copy_labels" src="https://github.com/user-attachments/assets/448252f2-e6a0-4bc9-ae30-42aa24aadc04" />
    </td>
  </tr>
</table>

### Connected component analysis
There are shortcut buttons for connected components labeling, keeping the largest cluster of connected labels, and to keep the largest fragment per label.

<img width="910" height="770" alt="conncomp" src="https://github.com/user-attachments/assets/8c7f41b2-fa58-48cc-8921-29e703195401" />

### Select / delete labels that overlap with a binary mask
All labels that share any pixel overlap with the mask are selected or removed.
<img width="1500" height="844" alt="select_delete" src="https://github.com/user-attachments/assets/c3d96516-d34d-468c-9c51-3fed4e904463" />

### Binary mask interpolation
It is possible to interpolate a 3D or 4D mask to fill in the region in between. In 3D, this means creating a 3D volume from slices, in 4D this means creating a time series of a volume that linearly 'morphs' into a different shape.

<table>
  <tr>
    <td width="40%">
      <img width="400" alt="interpolate" src="https://github.com/user-attachments/assets/74115915-936a-48cb-a6b3-415d09a9ed48" />
    </td>
    <td width="60%">
      <img width="600" alt="labelinterpolation gif" src="https://github.com/user-attachments/assets/c7384096-478d-448e-882c-691d568e83f8" />
    </td>
  </tr>
</table>

### Measuring label properties
You can measure label properties, including intensity (if a matching image layer is provided), area/volume, perimeter/surface area, circularity/sphericity, ellipse/ellipsoid axes in the 'Region Properties' tab. The plugin uses scikit-image regionprops with extended properties for 3D shapes based on methods from PoreSpy.
Make sure you set the dimensions correctly in the 'Extra layer controls' tab, to distinguish between measuring in 2D + time, 3D, and 3D + time, depending on your layer dimensions (2D to 4D). Once finished, a table displays the measurements, and a filter widget allows you to select objects matching a condition. The measurements are also displayed in the 'Plot'-tab for each layer for which you ran the region properties calculation.

![table_widget](https://github.com/user-attachments/assets/a7cd7686-7f90-46de-9864-0cac3aa77cf2)

## See also
This plugin has taken inspiration from other napari plugins with similar and more advanced functionalities for measuring features, such as:

- ![napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)
- ![napari-pyclesperanto-assistant](https://napari-hub.org/plugins/napari-pyclesperanto-assistant)
- ![morphometrics](https://napari-hub.org/plugins/morphometrics)
- ![napari-clusters-plotter](https://napari-hub.org/plugins/napari-clusters-plotter)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-segmentation-toolbox" is free and open source software

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

[file an issue]: https://github.com/AnniekStok/napari-segmentation-toolbox/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
