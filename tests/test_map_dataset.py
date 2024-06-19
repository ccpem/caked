from __future__ import annotations

from pathlib import Path

import testdata_mrc
import testdata_npy
import torch

from caked.dataloader import MapDataLoader, MapDataset

ORIG_DIR = Path.cwd()
TEST_DATA_MRC = Path(testdata_mrc.__file__).parent.joinpath("mrc")
TEST_DATA_NPY = Path(testdata_npy.__file__).parent


DISK_CLASSES_NONE = None
DATATYPE_MRC = "mrc"
VOXNORM = "voxnorm"
NORM = "norm"
MASKCROP = "maskcrop"
PADDING = "padding"
TRANSFORM_ALL = [VOXNORM, NORM, PADDING]


def test_map_dataloader():
    test_loader = MapDataLoader()

    assert test_loader is not None
    assert isinstance(test_loader, MapDataLoader)


def test_map_dataset():
    print()
    test_map_dataset = MapDataset(path=next(TEST_DATA_MRC.glob(f"*{DATATYPE_MRC}")))
    assert test_map_dataset is not None
    assert isinstance(test_map_dataset, MapDataset)


def test_slices():
    test_map_dataset = MapDataset(
        path=next(TEST_DATA_MRC.glob(f"*{DATATYPE_MRC}")), transforms=[], augments=[]
    )
    slice_, _, _ = test_map_dataset.__getitem__(0)

    assert isinstance(slice_, torch.Tensor)
    assert len(test_map_dataset) == 2
    assert slice_.shape == (49, 46, 48)


def test_transforms():
    test_map_dataset = MapDataset(
        path=next(TEST_DATA_MRC.glob(f"*{DATATYPE_MRC}")),
        transforms=TRANSFORM_ALL,
        augments=[],
    )
    slice_, _, _ = test_map_dataset.__getitem__(0)

    assert len(test_map_dataset) == 8
    assert slice_.shape == (64, 64, 64)


