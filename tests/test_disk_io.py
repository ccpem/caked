from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests import testdata_mrc

from caked.dataloader import DiskDataLoader, DiskDataset

ORIG_DIR = Path.cwd()
TEST_DATA_MRC = Path(testdata_mrc.__file__).parent
DISK_PIPELINE = "disk"
DATASET_SIZE_ALL = None
DATASET_SIZE_SOME = 3
DISK_CLASSES_FULL = ["1b23", "1dfo", "1dkg", "1e3p"]
DISK_CLASSES_SOME = ["1b23", "1dkg"]
DISK_CLASSES_MISSING = ["2b3a", "1b23"]
DISK_CLASSES_NONE = None
DATATYPE_MRC = "mrc"


def test_class_instantiation():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_SOME,
        dataset_size=DATASET_SIZE_SOME,
        save_to_disk=False,
        training=True,
    )
    assert isinstance(test_loader, DiskDataLoader)
    assert test_loader.pipeline == DISK_PIPELINE


def test_dataset_instantiation():
    test_dataset = DiskDataset(paths=["test"])
    assert isinstance(test_dataset, DiskDataset)


def test_load_dataset_no_classes():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE, classes=DISK_CLASSES_NONE, dataset_size=DATASET_SIZE_ALL
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL))


def test_load_dataset_all_classes():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE, classes=DISK_CLASSES_FULL, dataset_size=DATASET_SIZE_ALL
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_FULL)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_FULL))


def test_load_dataset_some_classes():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE, classes=DISK_CLASSES_SOME, dataset_size=DATASET_SIZE_ALL
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    assert isinstance(test_loader.dataset, DiskDataset)
    assert len(test_loader.classes) == len(DISK_CLASSES_SOME)
    assert all(a == b for a, b in zip(test_loader.classes, DISK_CLASSES_SOME))


def test_load_dataset_missing_class():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_MISSING,
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
    assert test_item_name in DISK_CLASSES_FULL
    assert isinstance(test_item_image, torch.Tensor)


def test_get_loader_training_false():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL,
        dataset_size=DATASET_SIZE_ALL,
        training=False,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    torch_loader = test_loader.get_loader(batch_size=64)
    assert isinstance(torch_loader, torch.utils.data.DataLoader)


def test_get_loader_training_true():
    test_loader = DiskDataLoader(
        pipeline=DISK_PIPELINE,
        classes=DISK_CLASSES_FULL,
        dataset_size=DATASET_SIZE_ALL,
        training=True,
    )
    test_loader.load(datapath=TEST_DATA_MRC, datatype=DATATYPE_MRC)
    torch_loader_train, torch_loader_val = test_loader.get_loader(
        split_size=0.8, batch_size=64
    )
    assert isinstance(torch_loader_train, torch.utils.data.DataLoader)
    assert isinstance(torch_loader_val, torch.utils.data.DataLoader)
