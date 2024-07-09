from __future__ import annotations

from pathlib import Path

import torch

from caked.dataloader import MapDataLoader, MapDataset
from caked.utils import duplicate_and_augment_from_hdf5

ORIG_DIR = Path.cwd()


DISK_CLASSES_NONE = None
DATATYPE_MRC = "mrc"
VOXNORM = "voxnorm"
NORM = "norm"
MASKCROP = "maskcrop"
PADDING = "padding"
ROTATION = "randrot"
TRANSFORM_ALL = [VOXNORM, NORM, PADDING]
AUGMENT_ALL = [ROTATION]


def test_map_dataloader():
    test_loader = MapDataLoader()

    assert test_loader is not None
    assert isinstance(test_loader, MapDataLoader)


def test_map_dataset(test_data_single_mrc_dir):
    test_map_dataset = MapDataset(
        path=next(test_data_single_mrc_dir.glob(f"*{DATATYPE_MRC}"))
    )
    assert test_map_dataset is not None
    assert isinstance(test_map_dataset, MapDataset)


def test_slices(test_data_single_mrc_dir):
    test_map_dataset = MapDataset(
        path=next(test_data_single_mrc_dir.glob(f"*{DATATYPE_MRC}")),
        transforms=[],
        augments=[],
    )
    slice_, _, _ = test_map_dataset.__getitem__(0)

    assert isinstance(slice_, torch.Tensor)
    assert len(test_map_dataset) == 2
    assert slice_.shape == (49, 46, 48)


def test_transforms(test_data_single_mrc_dir):
    test_map_dataset = MapDataset(
        path=next(test_data_single_mrc_dir.glob(f"*{DATATYPE_MRC}")),
        transforms=TRANSFORM_ALL,
        augments=[],
    )
    test_map_dataset.load_map_objects()
    test_map_dataset.transform()
    slice_, _, _ = test_map_dataset.__getitem__(0)

    assert len(test_map_dataset) == 8
    assert slice_.shape == (64, 64, 64)


def test_dataloader_load_to_HDF5_file(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader()
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )

    assert test_map_dataloader is not None
    assert isinstance(test_map_dataloader, MapDataLoader)
    assert test_map_dataloader.dataset is not None
    assert test_data_single_mrc_temp_dir.joinpath("raw_map_data.h5").exists()


def test_dataloader_load_to_HDF5_file_with_transforms(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader(
        transformations=TRANSFORM_ALL,
    )
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )

    assert test_map_dataloader is not None
    assert isinstance(test_map_dataloader, MapDataLoader)
    assert test_map_dataloader.dataset is not None
    assert test_data_single_mrc_temp_dir.joinpath("raw_map_data.h5").exists()


def test_add_duplicate_dataset_to_dataloader(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader(
        transformations=TRANSFORM_ALL,
    )
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )

    duplicate_and_augment_from_hdf5(
        test_map_dataloader,
        ids=[
            next(test_data_single_mrc_temp_dir.glob(f"*{DATATYPE_MRC}")).stem,
            next(test_data_single_mrc_temp_dir.glob(f"*{DATATYPE_MRC}")).stem,
        ],
    )
    hdf5_store = test_map_dataloader.dataset.datasets[0].map_hdf5_store

    assert len(hdf5_store.keys()) == 3
    assert "realmap_map" in hdf5_store.keys()  # noqa: SIM118
    assert "1--realmap_map" in hdf5_store.keys()  # noqa: SIM118
    assert "2--realmap_map" in hdf5_store.keys()  # noqa: SIM118


def test_add_duplicate_dataset_to_dataloader_with_augments(
    test_data_single_mrc_temp_dir,
):
    test_map_dataloader = MapDataLoader(
        transformations=TRANSFORM_ALL,
    )
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
    )
    duplicate_and_augment_from_hdf5(
        ids=[next(test_data_single_mrc_temp_dir.glob(f"*{DATATYPE_MRC}")).stem],
        map_data_loader=test_map_dataloader,
        augmentations=AUGMENT_ALL,
    )
    hdf5_store = test_map_dataloader.dataset.datasets[0].map_hdf5_store
    assert len(hdf5_store.keys()) == 2
    assert "realmap_map" in hdf5_store.keys()  # noqa: SIM118
    assert "1--realmap_map" in hdf5_store.keys()  # noqa: SIM118

    assert len(test_map_dataloader.dataset.datasets[0]) == 8
    assert len(test_map_dataloader.dataset.datasets[1]) == 8

    assert len(test_map_dataloader.dataset) == 16


def test_dataloader_load_multi_process(test_data_single_mrc_temp_dir):
    test_map_dataloader = MapDataLoader()
    test_map_dataloader.load(
        datapath=test_data_single_mrc_temp_dir,
        datatype=DATATYPE_MRC,
        num_workers=2,
    )

    assert test_map_dataloader is not None
    assert isinstance(test_map_dataloader, MapDataLoader)
    assert test_map_dataloader.dataset is not None
    assert test_data_single_mrc_temp_dir.joinpath("raw_map_data.h5").exists()

    # test_map_dataloader.
