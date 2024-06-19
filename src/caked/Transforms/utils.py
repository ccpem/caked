from __future__ import annotations

import math

import numpy as np
from ccpem_utils.map.parse_mrcmapobj import MapObjHandle


def pad_map_grid_sample(
    mapobj: MapObjHandle,
    ext_dim: tuple,
    inplace: bool = False,
    fill_padding: float | None = None,
    left: bool = True,
) -> MapObjHandle | None:
    """Takes an input map object and pads it with zeros to the specified extent.

    :param mapobj: (MapObjHandle) map object to be padded
    :param ext_dim: (tuple) the extent of the padding in each dimension (X, Y, Z)
    :param inplace: (bool) whether to modify the input map object or return a new one
    :param fill_padding: (float) value to fill the padding with
    :param left: (bool) if there is an odd number of slices to pad, whether to pad more on the left or right

    :return: (MapObjHandle) the padded map object
    """

    def even_odd_split(n):
        if n % 2 == 0:
            return n // 2, n // 2

        return n // 2, n - n // 2

    nx, ny, nz = ext_dim[::-1]
    nx1, nx2 = even_odd_split(nx)
    ny1, ny2 = even_odd_split(ny)
    nz1, nz2 = even_odd_split(nz)

    padded_array = pad_array_numpy(
        mapobj.data, nx1, nx2, ny1, ny2, nz1, nz2, fill_padding=fill_padding, left=left
    )
    # the start is at the base of the xyz grid
    # I want to move the origin to the base of the padded grid
    start = (nx1, ny1, nz1)

    ox = mapobj.origin[0] - start[0] * mapobj.apix[0]
    oy = mapobj.origin[1] - start[1] * mapobj.apix[1]
    oz = mapobj.origin[2] - start[2] * mapobj.apix[2]
    nstx = mapobj.nstart[0] - start[0]
    nsty = mapobj.nstart[1] - start[1]
    nstz = mapobj.nstart[2] - start[2]
    if not inplace:
        newmap = mapobj.copy()
        newmap.origin = (ox, oy, oz)
        newmap.data = padded_array
        newmap.nstart = (nstx, nsty, nstz)
        newmap.update_header_by_data()
        return newmap

    mapobj.origin = (ox, oy, oz)
    mapobj.data = padded_array
    mapobj.nstart = (nstx, nsty, nstz)
    mapobj.update_header_by_data()

    return None


def pad_array_numpy(arr, nx1, nx2, ny1, ny2, nz1, nz2, fill_padding=None, left=True):
    """

    Pad an array with specified increments along each dimension.
    Arguments:
        *nx,ny,nz*
           Number of slices to add to either sides of each dimension.
    Return:
        array
    """

    # the nx, ny, nz values should be the total number of slices to add, split as evenly as possible

    if not left:
        nx1, nx2 = nx2, nx1
        ny1, ny2 = ny2, ny1
        nz1, nz2 = nz2, nz1

    return np.pad(
        arr,
        ((nz1, nz2), (ny1, ny2), (nx1, nx2)),
        mode="constant",
        constant_values=fill_padding,
    )


def mask_from_labelobj(label_mapobj: MapObjHandle):
    """
    Create a mask from a label object, where the mask is a boolean array
    where the values are 1 for the labels and 0 for the background.
    """
    mask_obj = label_mapobj.copy(deep=True)
    arr = mask_obj.data
    arr[arr > 1] = 1
    arr[arr < 0] = 0
    mask_obj.data = arr
    return mask_obj


def divx(x, d=8):
    """Ensure the number is divisible (to an integer) by x (to ensure it can pool
    and concatenate max 3 times (2^3))."""
    if x % d != 0:
        y = math.ceil(x / d)
        x = y * d
    return x
