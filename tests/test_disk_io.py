from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tests import testdata_mrc, testdata_npy

from caked.dataloader import DiskDataLoader, DiskDataset

ORIG_DIR = Path.cwd()
TEST_DATA_MRC = Path(testdata_mrc.__file__).parent
TEST_DATA_NPY = Path(testdata_npy.__file__).parent
DISK_PIPELINE = "disk"
DATASET_SIZE_ALL = None
DATASET_SIZE_SOME = 3
DISK_CLASSES_FULL_MRC = ["1b23", "1dfo", "1dkg", "1e3p"]
DISK_CLASSES_SOME_MRC = ["1b23", "1dkg"]
DISK_CLASSES_MISSING_MRC = ["2b3a", "1b23"]
DISK_CLASSES_FULL_NPY = ["2", "5", "a", "d", "e", "i", "j", "l", "s", "u", "v", "x"]
DISK_CLASSES_SOME_NPY = ["2", "5"]
DISK_CLASSES_MISSING_NPY = ["2", "a", "1"]

DISK_CLASSES_NONE = None
DATATYPE_MRC = "mrc"
DATATYPE_NPY = "npy"
TRANSFORM_ALL = "normalise,gaussianblur,shiftmin"
TRANSFORM_SOME = "normalise,gaussianblur"


def test_class_instantiation():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_SOME_MRC,
        dataset_size=DATASET_SIZE_SOME,
        save_to_disk=False,
        training=True,
    )
    assert isinstance(test_loader, DiskDataLoader)
    assert test_loader.pipeline == DISK_PIPELINE


def test_dataset_instantiation_mrc():
    test_dataset = DiskDataset(paths=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_dataset, DiskDataset)


def test_dataset_instantiation_npy():
    test_dataset = DiskDataset(paths=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_dataset, DiskDataset)


def test_load_dataset_no_classes():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE, classes=DISK_CLASSES_NONE, dataset_size=DATASET_SIZE_ALL
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL_MRC)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL_MRC))


def test_load_dataset_all_classes_mrc():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL_MRC)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL_MRC))


def test_load_dataset_all_classes_npy():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_NPY,
        dataset_size=DATASET_SIZE_ALL,
    )
    test_loader.load(datapath=TEST_DATA_NPY, datatype=DATATYPE_NPY)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL_NPY)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL_NPY))


def test_load_dataset_some_classes():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_SOME_MRC,
        dataset_size=DATASET_SIZE_ALL,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_SOME_MRC)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_SOME_MRC))


def test_load_dataset_missing_class():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_MISSING_MRC,
        dataset_size=DATASET_SIZE_ALL,
    )
    with pytest.raises(Exception, match=r".*Missing classes: .*"):
        test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)


def test_one_image():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE, classes=DISK_CLASSES_NONE, dataset_size=DATASET_SIZE_ALL
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    test_dataset = test_loader.dataset
    test_item_image, test_item_name = test_dataset.__getitem__(1)
    assert test_item_name in DISK_CLASSES_FULL_MRC
    assert isinstance(test_item_image, torch.Tensor)


def test_get_loader_training_false():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=False,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    torch_loader = test_loader.get_loader(batch_size=64)
    assert isinstance(torch_loader, torch.utils.data.DataLoader)


def test_get_loader_training_true():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    torch_loader_train, torch_loader_val = test_loader.get_loader(
        split_size=0.8, batch_size=64
    )
    assert isinstance(torch_loader_train, torch.utils.data.DataLoader)
    assert isinstance(torch_loader_val, torch.utils.data.DataLoader)


def test_get_loader_training_fail():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    with pytest.raises(Exception, match=r".* sets must be larger than .*"):
        torch_loader_train, torch_loader_val = test_loader.get_loader(
            split_size=1, batch_size=64
        )


def test_processing_data_all_transforms():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_MRC,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
        transformations=TRANSFORM_ALL,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert test_loader.dataset.normalise
    assert test_loader.dataset.shiftmin
    assert test_loader.dataset.gaussianblur
    image, label = next(iter(test_loader.dataset))
    image = np.squeeze(image.cpu().numpy())
    assert len(image[0]) == len(image[1]) == len(image[2])
    assert label in DISK_CLASSES_FULL_MRC


def test_processing_data_some_transforms_npy():
    test_loader_transf = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_NPY,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
        transformations=TRANSFORM_SOME,
    )
    test_loader_none = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL_NPY,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader_none.load(datapath=TEST_DATA_NPY, datatype=DATATYPE_NPY)
    test_loader_transf.load(datapath=TEST_DATA_NPY, datatype=DATATYPE_NPY)
    assert test_loader_transf.dataset.normalise
    assert not test_loader_transf.dataset.shiftmin
    assert test_loader_transf.dataset.gaussianblur
    image_none, label_none = next(iter(test_loader_none.dataset))
    image_none = np.squeeze(image_none.cpu().numpy())
    assert len(image_none[0]) == len(image_none[1])
    assert label_none in DISK_CLASSES_FULL_NPY
    image_transf, label_transf = next(iter(test_loader_transf.dataset))
    image_transf = np.squeeze(image_transf.cpu().numpy())
    assert len(image_transf[0]) == len(image_transf[1])
    assert label_transf in DISK_CLASSES_FULL_NPY
    assert len(image_none[0]) == len(image_transf[0])
    assert len(image_none[1]) == len(image_transf[1])
