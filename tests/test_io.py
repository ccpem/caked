from __future__ import annotations

from caked.dataloader import DiskDataLoader


def test_class_instantiation():
    test_loader = DiskDataLoader(
        pipeline="test",
        classes=["test"],
        dataset_size=3,
        save_to_disk=False,
        training=True,
    )
    assert isinstance(test_loader, DiskDataLoader)
