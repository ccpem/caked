from __future__ import annotations

from enum import Enum

import numpy as np
from ccpem_utils.map.mrc_map_utils import (
    interpolate_to_grid,
    normalise_mapobj,
    pad_map_grid_split_distribution,
)
from ccpem_utils.map.parse_mrcmapobj import MapObjHandle

from .base import TransformBase
from .utils import divx, mask_from_labelobj


class Transforms(Enum):
    """
    Enum class for transformations.

    """

    VOXNORM = "voxnorm"
    NORM = "norm"
    MASKCROP = "maskcrop"
    PADDING = "padding"


def get_transform(transform: str) -> TransformBase:
    """
    Get the transformation object.

    :param transform: (str) transformation to apply

    :return: (MapObjHandle) transformed MapObjHandle
    """

    if transform == Transforms.VOXNORM.value:
        return MapObjectVoxelNormalisation()
    if transform == Transforms.NORM.value:
        return MapObjectNormalisation()
    if transform == Transforms.MASKCROP.value:
        return MapObjectMaskCrop()
    if transform == Transforms.PADDING.value:
        return MapObjectPadding()
    msg = f"Unknown transform: {transform}, please choose from {Transforms.__members__}"
    raise ValueError(msg)


class ComposeTransform:
    """
    Compose multiple transformations together.

    :param transforms: (list) list of transformations to compose

    :return: (dict) transformed MapObjHandle kwargs
    """

    def __init__(self, transforms: list[str]):
        self.transforms = transforms

    def __call__(self, *args: list[MapObjHandle | None], **kwargs) -> dict:
        for transform in self.transforms:
            for mapobj in args:
                if mapobj is None:
                    continue  # type: ignore[unreachable]

                _, kwargs = get_transform(transform)(mapobj, **kwargs)

        return kwargs


class DecomposeToSlices:
    """
    Decomposes a 3D map into smaller 3D slices.
    """

    def __init__(self, map_shape: tuple, **kwargs):
        step = kwargs.get("step", 1)
        cshape = kwargs.get("cshape", 1)
        slices, slice_indicies = [], []

        for i in range(0, map_shape[0], step):
            for j in range(0, map_shape[1], step):
                for k in range(0, map_shape[2], step):
                    if (
                        i + cshape > map_shape[0]
                        or j + cshape > map_shape[1]
                        or k + cshape > map_shape[2]
                    ):
                        continue
                    slices.append(
                        (
                            slice(i, i + cshape),
                            slice(j, j + cshape),
                            slice(k, k + cshape),
                        )
                    )
                    slice_indicies.append((i, j, k))

        if len(slice_indicies) == 0:
            msg = ("No slices were generated, please check the step and "
                   "cshape values. Using single slice.")
            print(msg)
            slices.append(
                (
                    slice(0, cshape),
                    slice(0, cshape),
                    slice(0, cshape),
                )
            )
            slice_indicies.append((0, 0, 0))
        self.slices = slices
        self.slice_indicies = slice_indicies

    def select_slices_min_class_fraction(
        self,
        label_array: np.ndarray,
        class_fraction: dict[int, float],
        padding: bool = True,
        padding_value: float = 0.0,
        skip_small: bool = False,
    ) -> list[np.ndarray]:
        """
        Select slices based on the fraction of classes.

        :param label_array: (np.ndarray) label array to check class values
        :param class_fraction: (dict[int, float]) fraction threshold
            for each class
        :param padding: (bool) whether to pad the slice if 
            it exceeds the map array dimensions
        :param padding_value: (float) value to use for padding
        :param skip_small: (bool) whether to skip small slices (that
            exceed the map array dimensions)
        """
        
        selected_slices = []
        selected_indices = []
        label_slices = []
        for idx, slc in enumerate(self.slices):
            # extract slice array
            slice_data = self.extract_slices(
                label_array,
                padding=padding,
                padding_value=padding_value,
                skip_small=skip_small,
            )
            # check class fractions
            filter_slice = False
            if slice_data:
                for class_value, fraction in class_fraction.items():
                    class_count = np.sum(slice_data == class_value)
                    total_count = slice_data.size
                    if total_count == 0:
                        break
                    if (class_count / total_count) < fraction:
                        filter_slice = True
                # if not remove slice
                if not filter_slice:
                    selected_slices.append(slc)
                    selected_indices.append(self.slice_indicies[idx])
                    label_slices.append(slice_data)
            self.slices = selected_slices
            self.slice_indicies = selected_indices
        return label_slices

    def extract_slices(
        self,
        map_array: np.ndarray,
        padding: bool = True,
        padding_value: float = 0.0,
        skip_small: bool = False,
    ) -> list[np.ndarray]:
        """
        Extract slices from a Map array.

        :param map_array: (np.ndarray) map array to extract slices from
        :param padding: (bool) whether to pad the slice if 
            it exceeds the map array dimensions
        :param padding_value: (float) value to use for padding
        :param skip_small: (bool) whether to skip small slices (that
            exceed the map array dimensions)

        :return: (list) list of np.ndarray slices
        """
        array_slices = []
        slices = []
        slice_indicies = []
        for idx, slc in enumerate(self.slices):
            slice_data = self.get_slice_array(
                map_array,
                slc,
                padding=padding,
                padding_value=padding_value,
                skip_small=skip_small,)
            if slice_data: 
                array_slices.append(slice_data)
                slices.append(slc)
                slice_indicies.append(self.slice_indicies[idx])
        # Update slices to only include successful extractions
        self.slices = slices
        self.slice_indicies = slice_indicies
        return array_slices

    def get_slice_array(
        map_array: np.ndarray,
        slc: tuple[slice, slice, slice],
        padding: bool = True,
        padding_value: float = 0.0,
        skip_small: bool = False,
    ) -> np.ndarray:
        """
        Get a specific slice from a Map array.

        :param map_array: (np.ndarray) map array to extract the slice from
        :param slc: (tuple) slice indices
        :param padding: (bool) whether to pad the slice if 
            it exceeds the map array dimensions
        :param padding_value: (float) value to use for padding
        :param skip_small: (bool) whether to skip small slices (that 
            exceed the map array dimensions)

        :return: (np.ndarray) extracted slice
        """
        if len(map_array.shape) != len(slc):
            msg = "Map array must have the same number of dimensions as the slice."
            raise ValueError(msg)
        if any(s.stop > map_array.shape[idx] for idx, s in enumerate(slc)): 
            print("Slice exceeds map array dimensions.")
            if padding:
                # Create a padded slice
                padding_width = [
                    (0, max(0, s.stop - map_array.shape[idx]))
                    for idx, s in enumerate(slc)
                ]
                slice_data = np.pad(
                    map_array[slc],
                    pad_width=padding_width,
                    mode="constant",
                    constant_values=padding_value,
                )
                return slice_data
            elif skip_small:
                return None
        slice_data = map_array[slc]
        return slice_data


class MapObjectVoxelNormalisation(TransformBase):
    """
    Resamples a map object to a desired voxel size if outside of vox_sh_min and
    vox_sh_max.

    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ) -> tuple[MapObjHandle, dict]:
        # This is needed to do the normalisation but I need to check if label obj is affected by this

        vox = kwargs.get("vox", 1.0)
        vox_min = kwargs.get("vox_min", 0.95)
        vox_max = kwargs.get("vox_max", 1.05)

        if not vox_min < vox < vox_max:
            msg = f"Voxel size must be within the range of {vox_min} and {vox_max}."
            raise ValueError(msg)

        voxx, voxy, voxz = mapobj.apix
        sample = np.array(mapobj.shape)
        if voxx > vox_max or voxx < vox_min:
            sample[2] = int(mapobj.dim[0] / vox)
        if voxy > vox_max or voxy < vox_min:
            sample[1] = int(mapobj.dim[1] / vox)
        if voxz > vox_max or voxz < vox_min:
            sample[0] = int(mapobj.dim[2] / vox)
        sample = tuple(sample)
        interpolate_to_grid(
            mapobj,
            sample,
            (vox, vox, vox),
            mapobj.origin,
            inplace=True,
            prefilter_input=mapobj.all_transforms,
        )

        mapobj.update_header_by_data()

        return mapobj, kwargs


class MapObjectNormalisation(TransformBase):
    """
    Normalise the voxel values of a Map Object.

    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ) -> tuple[MapObjHandle, dict]:
        if not mapobj.all_transforms:
            return mapobj, kwargs
        normalise_mapobj(
            mapobj,
            inplace=True,
        )

        return mapobj, kwargs


class MapObjectMaskCrop(TransformBase):
    """
    Crop a Map Object using a mask.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ) -> tuple[MapObjHandle, dict]:
        mask = kwargs.get("mask", None)
        if mask is None:
            msg = "Please provide a mask to crop the map object."
            raise ValueError(msg)

        mask = mask_from_labelobj(mask)

        return mapobj, kwargs


class MapObjectPadding(TransformBase):
    """
    Pad a Map Object.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ) -> tuple[MapObjHandle, dict]:
        ext_dim = [divx(d, kwargs.get("step", 1)) - d for d in mapobj.shape]

        left = kwargs.get("left", True)
        pad_map_grid_split_distribution(
            mapobj,
            ext_dim=ext_dim,
            fill_padding=0.0,
            left=left,
            inplace=True,
        )
        return mapobj, kwargs


# def data_scale(mapobj: MapObjHandle, desired_shape: tuple, inplace=True):
#     """
#     Resamples image to desired shape.

#     :param mapobj: (MapObjHandle) map object
#     :param desired_shape: (tuple(int, int, int)) desired shape
#     :param inplace: (bool) perform operation in place
#     :return: mapobj: (MapObjHandle) updated map object
#     """
#     interpolate_to_grid(mapobj, desired_shape, mapobj.apix, mapobj.origin, inplace=True)
#     if not inplace:
#         return mapobj

#     mapobj.update_header_by_data()
