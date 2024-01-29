from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader, Dataset


class AbstractDataLoader(DataLoader, ABC):
    @abstractmethod
    def read(self, pipeline, classes, volume_size, dataset_size, save_to_disk):
        pass

    @abstractmethod
    def process(self, dataset, classes, split_size, batch_size, training):
        pass


class AbstractDataset(ABC, Dataset):
    def __init__(self, origin: str, classes: Path) -> None:
        pass

    def __len__(self) -> int:
        pass

    @abstractmethod
    def set_len(self, length:int):
        pass

    @abstractmethod
    def augment(self, augment:bool, aug_type:str):
        pass
    