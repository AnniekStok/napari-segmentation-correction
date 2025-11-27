import math

import numpy as np
import pandas as pd
from skimage.measure import marching_cubes, mesh_surface_area, regionprops
from skimage.measure._regionprops import RegionProperties


class ExtendedRegionProperties(RegionProperties):
    """Adding additional properties to skimage.measure._regionprops following the logic from the porespy package with some modifications to include the spacing information."""

    @property
    def ellipse_axes(self):
        return (self.axis_major_length, self.axis_minor_length)

    @property
    def ellipsoid_axes(self):
        """
        Calculate the three axes radii of the fitted ellipsoid.

        This method calculates the principal axes of inertia for the label region, which are used to determine the lengths of the axes radii of an ellipsoid that approximates the shape of the region.

        Returns:
            tuple: A tuple containing the lengths of the three principal axes radii (longr, midr, shortr).

        """

        # Extract the coordinates of the region's voxels
        cell = np.where(self.image)
        voxel_count = self.voxel_count

        z, y, x = cell

        # Center the coordinates and apply spacing
        z = (z - np.mean(z)) * self._spacing[0]
        y = (y - np.mean(y)) * self._spacing[1]
        x = (x - np.mean(x)) * self._spacing[2]

        # Calculate the elements of the inertia tensor
        i_xx = np.sum(y**2 + z**2)
        i_yy = np.sum(x**2 + z**2)
        i_zz = np.sum(x**2 + y**2)
        i_xy = np.sum(x * y)
        i_xz = np.sum(x * z)
        i_yz = np.sum(y * z)
        i = np.array([[i_xx, -i_xy, -i_xz], [-i_xy, i_yy, -i_yz], [-i_xz, -i_yz, i_zz]])

        # Compute the eigenvalues and eigenvectors of the inertia tensor. The eigenvalues of the inertia tensor represent the principal moments of inertia, and the eigenvectors represent the directions of the principal axes.
        eig = np.linalg.eig(i)
        eigval = eig[0]

        # Identify the principal axes
        longaxis = np.where(np.min(eigval) == eigval)[0][0]
        shortaxis = np.where(np.max(eigval) == eigval)[0][0]
        midaxis = (
            0
            if shortaxis != 0 and longaxis != 0
            else 1
            if shortaxis != 1 and longaxis != 1
            else 2
        )

        # Calculate the lengths of the principal axes
        longr = math.sqrt(
            5.0
            / 2.0
            * (eigval[midaxis] + eigval[shortaxis] - eigval[longaxis])
            / voxel_count
        )
        midr = math.sqrt(
            5.0
            / 2.0
            * (eigval[shortaxis] + eigval[longaxis] - eigval[midaxis])
            / voxel_count
        )
        shortr = math.sqrt(
            5.0
            / 2.0
            * (eigval[longaxis] + eigval[midaxis] - eigval[shortaxis])
            / voxel_count
        )

        return (longr, midr, shortr)

    @property
    def circularity(self):
        """
        Calculate the circularity of the region.

        Circularity is defined as 4π * (Area / Perimeter^2).

        Returns:
            float: The circularity of the region.
        """
        return 4 * math.pi * self.area / self.perimeter**2

    @property
    def pixel_count(self):
        """
        Get the number of pixels in the region.

        Returns:
            int: The number of pixels in the region.
        """
        return self.voxel_count

    @property
    def surface_area(self):
        """
        Calculate the surface area of the region using the marching cubes algorithm.

        The marching cubes algorithm is used to extract a 2D surface mesh from a 3D volume.
        The mesh surface area is calculated with skimage.measure.mesh_surface_area.

        Returns:
            float: The surface area of the region.
        """
        verts, faces, _, _ = marching_cubes(
            self.image, level=0.5, spacing=self._spacing
        )
        surface_area = mesh_surface_area(verts, faces)
        return surface_area

    @property
    def sphericity(self):
        """
        Calculate the sphericity of the region.

        Sphericity is defined as the ratio of the surface area of a sphere with the same volume as the region to the surface area of the region.

        Returns:
            float: The sphericity of the region.
        """
        vol = self.volume
        r = (3 / 4 / np.pi * vol) ** (1 / 3)
        a_equiv = 4 * np.pi * r**2
        a_region = self.surface_area
        return a_equiv / a_region

    @property
    def volume(self):
        """
        Calculate the volume of the region.

        The volume is calculated as the number of voxels in the region multiplied by the product of the spacing in each dimension.

        Returns:
            float: The volume of the region.
        """
        vol = np.sum(self.image) * np.prod(self._spacing)
        return vol

    @property
    def voxel_count(self):
        """
        Get the number of voxels in the region.

        Returns:
            int: The number of voxels in the region.
        """
        voxel_count = int(np.sum(self.image))
        return voxel_count


def regionprops_extended(
    img: np.ndarray, spacing: tuple[float], intensity_image: np.ndarray | None = None
) -> list[ExtendedRegionProperties]:
    """
    Create instances of ExtendedRegionProperties that extend skimage.measure.RegionProperties.

    Args:
        img (np.ndarray): The labeled image.
        spacing (tuple[float]): The spacing between voxels in each dimension.
        intensity_image (np.ndarray, optional): The intensity image.

    Returns:
        list[ExtendedRegionProperties]: A list of ExtendedRegionProperties instances.
    """
    results = regionprops(img, intensity_image=intensity_image, spacing=spacing)
    for i, _ in enumerate(results):
        a = results[i]
        b = ExtendedRegionProperties(
            slice=a.slice,
            label=a.label,
            label_image=a._label_image,
            intensity_image=a._intensity_image,
            cache_active=a._cache_active,
            spacing=a._spacing,
        )
        results[i] = b

    return results


def props_to_dataframe(regionprops, selected_properties=None) -> pd.DataFrame:
    """Convert ExtendedRegionProperties instance to pandas dataframe, following the logic from porespy.metrics._regionprops.props_to_dataframe"""

    if selected_properties is None:
        selected_properties = regionprops[0].__dict__()

    new_props = ["label"]
    # need to check if any of the props return multiple values
    for item in selected_properties:
        if isinstance(getattr(regionprops[0], item), tuple | np.ndarray | list):
            for i, _ in enumerate(getattr(regionprops[0], item)):
                new_props.append(item + "-" + str(i + 1))
        else:
            new_props.append(item)

    # get the measurements for all properties of interest
    d = {}
    for k in new_props:
        if "-" in k:
            # If property is a tuple, extract the tuple element
            prop_name, idx = k.split("-")
            idx = int(idx) - 1
            d[k] = np.array([getattr(r, prop_name)[idx] for r in regionprops])
        else:
            d[k] = np.array([getattr(r, k) for r in regionprops])

    # Create pandas data frame an return
    df = pd.DataFrame(d)

    return df


def calculate_extended_props(
    image: np.ndarray,
    properties: list[str],
    spacing: list[float],
    intensity_image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create regionproperties, and convert to pandas dataframe"""

    props = regionprops_extended(
        image, spacing=spacing, intensity_image=intensity_image
    )
    return props_to_dataframe(props, properties) if len(props) > 0 else pd.DataFrame()
