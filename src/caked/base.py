from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader, Dataset


class AbstractDataLoader(ABC):
    def __init__(self, pipeline: str, classes: list[str], dataset_size: int, save_to_disk: bool, training: bool):
        self.pipeline = pipeline
        self.classes = classes
        self.dataset_size = dataset_size
        self.save_to_disk = save_to_disk
        self.training = training

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def get_loader(self, split_size: float, batch_size: int):
        pass


class AbstractDataset(ABC):
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
