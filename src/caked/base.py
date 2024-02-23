from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class AbstractDataLoader(ABC):
    """
    Abstract base class for data loaders.

    Args:
        pipeline (str): The pipeline to be used for data loading.
        classes (list[str]): The list of classes for the dataset.
        save_to_disk (bool): Whether to save the dataset to disk.
        training (bool): Whether the data loader is used for training.
        dataset_size (int, optional): The size of the dataset. Defaults to None.
    """

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
        """
        Load data of a given type from a given path.

        Args:
            datapath (str): The path to the data.
            datatype (str): The type of data to load.

        """

    @abstractmethod
    def process(self, paths, datatype):
        """
        Process the given paths with the specified datatype, according to transformations.

        Args:
            paths (list): A list of paths to be processed.
            datatype (str): The datatype to be used for processing.
        """

    @abstractmethod
    def get_loader(
        self,
        batch_size: int,
        split_size: float | None = None,
    ):
        """
        Returns a data loader object.

        Args:
            batch_size (int): The batch size for the data loader.
            split_size (float | None, optional): The split size for the data loader. Defaults to None.

        Returns:
            DataLoader: The data loader object.
        """


class AbstractDataset(ABC, Dataset):
    """
    An abstract base class for datasets.

    This class provides a blueprint for creating datasets that can be used with data loaders.
    Subclasses must implement the `augment` method.

    Attributes:
        None

    Methods:
        augment: Apply data augmentation to the dataset.

    """

    @abstractmethod
    def augment(self, augment: bool, aug_type: str):
        pass
