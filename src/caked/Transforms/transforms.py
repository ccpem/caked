from __future__ import annotations

from enum import Enum

from ccpem_utils.map.mrc_map_utils import (
    crop_map_grid,
    normalise_mapobj,
)
from ccpem_utils.map.parse_mrcmapobj import MapObjHandle
from mlproteintoolbox.proteins.map_utils import voxel_normalisation

from .base import MapObjTransformBase
from .utils import divx, mask_from_labelobj, pad_map_grid_sample


class Transforms(Enum):
    """ """

    VOXNORM = "voxnorm"
    NORM = "norm"
    MASKCROP = "maskcrop"
    PADDING = "padding"


def get_transform(transform: str) -> MapObjTransformBase:
    """ """

    if transform == Transforms.VOXNORM.value:
        return MapObjectVoxelNormalisation()
    if transform == Transforms.NORM.value:
        return MapObjectNormalisation()
    if transform == Transforms.MASKCROP.value:
        return MapObjectMaskCrop()
    if transform == Transforms.PADDING.value:
        return MapObjectPadding()
    msg = f"Unknown transform: {transform}"
    raise ValueError(msg)


class ComposeTransform:
    """
    Compose multiple transformations together.

    :param transforms: (list) list of transformations to compose

    :return: (MapObjHandle) transformed MapObjHandle
    """

    def __init__(self, transforms: list[str]):
        self.transforms = transforms

    def __call__(self, mapobj: MapObjHandle, **kwargs) -> MapObjHandle:
        for transform in self.transforms:
            mapobj = get_transform(transform)(mapobj, **kwargs)
            if transform == Transforms.MASKCROP.value:
                kwargs["ext_dim"] = [
                    divx(d, kwargs.get("step", 1)) for d in mapobj.shape
                ]
        return kwargs


class DecomposeToSlices:
    """ """

    def __init__(self, mapobj: MapObjHandle, **kwargs):
        step = kwargs.get("step", 1)
        cshape = kwargs.get("cshape", 1)
        slices, tiles = [], []
        for i in range(0, mapobj.data.shape[0], step):
            for j in range(0, mapobj.data.shape[1], step):
                for k in range(0, mapobj.data.shape[2], step):
                    slices.append(
                        (
                            slice(i, i + cshape),
                            slice(j, j + cshape),
                            slice(k, k + cshape),
                        )
                    )
                    tiles.append((i, j, k))

        self.slices = slices
        self.tiles = tiles


class MapObjectVoxelNormalisation(MapObjTransformBase):
    """ """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ):
        norm_vox = kwargs.get("vox", None)
        norm_vox_lim = kwargs.get("vox_lim", None)

        voxel_normalisation(
            mapobj,
            vox=norm_vox,
            vox_min=norm_vox_lim[0],
            vox_max=norm_vox_lim[1],
            inplace=True,
        )

        return mapobj


class MapObjectNormalisation(MapObjTransformBase):
    """
    Normalise the voxel values of a 3D volume.

    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ):
        normalise_mapobj(
            mapobj,
            inplace=True,
        )

        return mapobj


class MapObjectMaskCrop(MapObjTransformBase):
    """
    Crop a Map Object using a mask.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ):
        mask = kwargs.get("mask", None)
        if mask is None:
            msg = "Please provide a mask to crop the map object."
            raise ValueError(msg)
        mask = mask_from_labelobj(mask)

        crop_map_grid(mapobj, input_maskobj=mask, inplace=True)

        return mapobj


class MapObjectPadding(MapObjTransformBase):
    """ """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        mapobj: MapObjHandle,
        **kwargs,
    ):
        ext_dim = kwargs.get("ext_dim", None)
        left = kwargs.get("left", True)

        pad_map_grid_sample(
            mapobj,
            ext_dim=ext_dim,
            fill_padding=0.0,
            left=left,
            inplace=True,
        )

        return mapobj
