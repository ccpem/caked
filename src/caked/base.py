from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class AbstractDataLoader(ABC):
    def __init__(
        self,
        pipeline: str,
        classes: list[str],
        save_to_disk: bool,
        training: bool,
        dataset_size: int | None = None,
    ):
        self.pipeline = pipeline
        self.classes = classes
        self.dataset_size = dataset_size
        self.save_to_disk = save_to_disk
        self.training = training

    @abstractmethod
    def load(self, datapath, datatype):
        pass

    @abstractmethod
    def process(self, paths, datatype):
        pass

    @abstractmethod
    def get_loader(
        self,
        batch_size: int,
        split_size: float | None = None,
    ):
        pass


class AbstractDataset(ABC, Dataset):
    @abstractmethod
    def augment(self, augment: bool, aug_type: str):
        pass
