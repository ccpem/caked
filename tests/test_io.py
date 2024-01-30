from src.caked.dataloader import DiskDataLoader
import pytest


def test_class_instantiation():
    test_loader = DiskDataLoader(pipeline="test", classes="test", dataset_size=3, save_to_disk=False, training=True)
    assert isinstance(test_loader, DiskDataLoader)