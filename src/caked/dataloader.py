"""
Functions for loading the data from disk.
Largely taken from https://github.com/alan-turing-institute/affinity-vae
"""

from __future__ import annotations

import logging
import os
import random
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mrcfile
import numpy as np
import torch
from ccpem_utils.map.parse_mrcmapobj import get_mapobjhandle
from scipy.ndimage import zoom
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms

from caked.hdf5_utils import HDF5DataStore
from caked.Transforms.augments import ComposeAugment
from caked.Transforms.transforms import ComposeTransform, DecomposeToSlices, Transforms

from .base import AbstractDataLoader, AbstractDataset

np.random.seed(42)
TRANSFORM_OPTIONS = ["normalise", "gaussianblur", "shiftmin"]


class DiskDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_size: int | None = None,
        save_to_disk: bool = False,
        training: bool = True,
        classes: list[str] | None = None,
        pipeline: str = "disk",
        transformations: list[str] | None = None,
    ) -> None:
        """
        DataLoader implementation for loading data from disk.

        Args:
            dataset_size (int | None, optional): The maximum number of samples to load from the dataset. If None, load all samples. Default is None.
            save_to_disk (bool, optional): Whether to save the loaded data to disk. Default is False.
            training (bool, optional): Whether the DataLoader is used for training. Default is True.
            classes (list[str] | None, optional): A list of classes to load from the dataset. If None, load all classes. Default is None.
            pipeline (str, optional): The data loading pipeline to use. Default is "disk".
            transformations (str | None, optional): The data transformations to apply. If None, no transformations are applied. Default is None.

        Raises:
            RuntimeError: If not all classes in the list are present in the directory.
            RuntimeError: If no processing is required because no transformations were provided.
            RuntimeError: If split size is not provided for training.
            RuntimeError: If train and validation sets are smaller than 2 samples.

        Attributes:
            dataset_size (int | None): The maximum number of samples to load from the dataset.
            save_to_disk (bool): Whether to save the loaded data to disk.
            training (bool): Whether the DataLoader is used for training.
            classes (list[str]): A list of classes to load from the dataset.
            pipeline (str): The data loading pipeline to use.
            transformations (str | None): The data transformations to apply.
            debug (bool): Whether to enable debug mode.
            dataset (DiskDataset): The loaded dataset.

        Methods:
            load(datapath, datatype): Load the data from the specified path and data type.
            process(paths, datatype): Process the loaded data with the specified transformations.
            get_loader(batch_size, split_size): Get the data loader for training or testing.
        """
        self.dataset_size = dataset_size
        self.save_to_disk = save_to_disk
        self.training = training
        self.pipeline = pipeline
        self.transformations = transformations
        self.debug = False

        if classes is None:
            self.classes = []
        else:
            self.classes = classes

    def load(self, datapath, datatype) -> None:
        """
        Load the data from the specified path and data type.

        Args:
            datapath (str): The path to the directory containing the data.
            datatype (str): The type of data to load.

        Returns:
            None
        """
        paths = [f for f in os.listdir(datapath) if "." + datatype in f]

        if not self.debug:
            random.shuffle(paths)

        # ids right now depend on the data being saved with a certain format (id in the first part of the name, separated by _)
        # TODO: make this more general/document in the README
        ids = np.unique([f.split("_")[0] for f in paths])
        if len(self.classes) == 0:
            self.classes = ids
        else:
            class_check = np.in1d(self.classes, ids)
            if not np.all(class_check):
                msg = "Not all classes in the list are present in the directory. Missing classes: {}".format(
                    np.asarray(self.classes)[~class_check]
                )
                raise RuntimeError(msg)
            class_check = np.in1d(ids, self.classes)
            if not np.all(class_check):
                logging.basicConfig(format="%(message)s", level=logging.INFO)
                logging.info(
                    "Not all classes in the directory are present in the "
                    "classes list. Missing classes: %s. They will be ignored.",
                    (np.asarray(ids)[~class_check]),
                )

        paths = [
            Path(datapath) / p
            for p in paths
            for c in self.classes
            if c in p.split("_")[0]
        ]
        if self.dataset_size is not None:
            paths = paths[: self.dataset_size]

        if self.transformations is None:
            self.dataset = DiskDataset(paths=paths, datatype=datatype)
        else:
            self.dataset = self.process(paths=paths, datatype=datatype)

    def process(self, paths: list[str], datatype: str):
        """
        Process the loaded data with the specified transformations.

        Args:
            paths (list[str]): List of file paths to the data.
            datatype (str): Type of data being processed.

        Returns:
            DiskDataset: Processed dataset object.

        Raises:
            RuntimeError: If no transformations were provided.
        """
        if self.transformations is None:
            msg = "No processing to do as no transformations were provided."
            raise RuntimeError(msg)
        transforms = list(self.transformations)
        rescale = 0
        normalise = False
        if "normalise" in transforms:
            normalise = True
            transforms.remove("normalise")

        gaussianblur = False
        if "gaussianblur" in transforms:
            gaussianblur = True
            transforms.remove("gaussianblur")

        shiftmin = False
        if "shiftmin" in transforms:
            shiftmin = True
            transforms.remove("shiftmin")

        for i in transforms:
            if i.startswith("rescale"):
                transforms.remove(i)
                rescale = int(i.split("=")[-1])

        if len(transforms) > 0:
            msg = f"The following transformations are not supported: {transforms}"
            raise RuntimeError(msg)

        return DiskDataset(
            paths=paths,
            datatype=datatype,
            rescale=rescale,
            normalise=normalise,
            gaussianblur=gaussianblur,
            shiftmin=shiftmin,
        )

    def get_loader(
        self,
        batch_size: int,
        split_size: float | None = None,
        no_val_drop: bool = False,
    ):
        """
        Retrieve the data loader.

        Args:
            batch_size (int): The batch size for the data loader.
            split_size (float | None, optional): The percentage of data to be used for validation set.
                If None, the entire dataset will be used for training. Defaults to None.
            no_val_drop (bool, optional): If True, the last batch of validation data will not be dropped if it is smaller than batch size. Defaults to False.

        Returns:
            DataLoader or Tuple[DataLoader, DataLoader]: The data loader(s) for testing or training/validation, according to whether training is True or False.

        Raises:
            RuntimeError: If split_size is None and the method is called for training.
            RuntimeError: If the train and validation sets are smaller than 2 samples.

        """
        if self.training:
            if split_size is None:
                msg = "Split size must be provided for training. "
                raise RuntimeError(msg)
            # split into train / val sets
            idx = np.random.permutation(len(self.dataset))
            if split_size < 1:
                split_size = split_size * 100

            s = int(np.ceil(len(self.dataset) * int(split_size) / 100))
            if s < 2:
                msg = "Train and validation sets must be larger than 1 sample, train: {}, val: {}.".format(
                    len(idx[:-s]), len(idx[-s:])
                )
                raise RuntimeError(msg)
            train_data = Subset(self.dataset, indices=idx[:-s])
            val_data = Subset(self.dataset, indices=idx[-s:])

            loader_train = DataLoader(
                train_data,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                drop_last=True,
            )
            loader_val = DataLoader(
                val_data,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                drop_last=(not no_val_drop),
            )
            return loader_train, loader_val

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )


class MapDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_size: int | None = None,
        save_to_disk: bool = False,
        training: bool = True,
        classes: list[str] | None = None,
        pipeline: str = "disk",
        transformations: list[str] | None = None,
        augmentations: list[str] | None = None,
    ) -> None:
        """
        DataLoader implementation for loading map data from disk.
        """
        self.dataset_size = dataset_size
        self.save_to_disk = save_to_disk
        self.training = training
        self.pipeline = pipeline
        self.transformations = transformations
        self.augmentations = augmentations
        self.debug = False
        self.classes = classes

        if self.classes is None:
            self.classes = []
        if self.transformations is None:
            self.transformations = []
        if self.augmentations is None:
            self.augmentations = []

    def __add__(self, other):
        if not isinstance(other, MapDataLoader):
            msg = "Can only add two MapDataLoader objects together."
            raise TypeError(msg)
        if self.pipeline != other.pipeline:
            msg = "Both MapDataLoader objects must use the same pipeline."
            raise ValueError(msg)
        if self.transformations != other.transformations:
            msg = "Both MapDataLoader objects must use the same transformations."
            raise ValueError(msg)
        if self.augmentations != other.augmentations:
            msg = "Both MapDataLoader objects must use the same augmentations."
            raise ValueError(msg)
        if self.classes != other.classes:
            msg = "Both MapDataLoader objects must use the same classes."
            raise ValueError(msg)
        if self.dataset_size != other.dataset_size:
            msg = "Both MapDataLoader objects must use the same dataset size."
            raise ValueError(msg)
        if self.save_to_disk != other.save_to_disk:
            msg = "Both MapDataLoader objects must use the same save to disk option."
            raise ValueError(msg)
        if self.training != other.training:
            msg = "Both MapDataLoader objects must use the same training option."
            raise ValueError(msg)

        new_loader = MapDataLoader(
            dataset_size=self.dataset_size,
            save_to_disk=self.save_to_disk,
            training=self.training,
            classes=self.classes,
            pipeline=self.pipeline,
            transformations=self.transformations,
            augmentations=self.augmentations,
        )
        new_loader.dataset = ConcatDataset([self.dataset, other.dataset])
        return new_loader

    def load(self, datapath, datatype, label_path=None, weight_path=None) -> None:
        """
        Load the data from the specified path and data type.

        Args:
            datapath (str): The path to the directory containing the data.
            datatype (str): The type of data to load.

        Returns:
            None
        """
        datapath = Path(datapath)
        label_path = Path(label_path) if label_path is not None else None
        weight_path = Path(weight_path) if weight_path is not None else None

        datasets = []
        num_workers = 6

        paths = list(datapath.rglob(f"*.{datatype}"))
        label_paths = (
            list(label_path.rglob(f"*.{datatype}")) if label_path is not None else None
        )
        weight_paths = (
            list(weight_path.rglob(f"*.{datatype}"))
            if weight_path is not None
            else None
        )

        if not self.debug:
            random.shuffle(paths)

        # ids right now depend on the data being saved with a certain format (id in the first part of the name, separated by _)
        # TODO: make this more general/document in the README
        ids = np.unique([file.name.split("_")[0] for file in paths])
        if len(self.classes) == 0:
            self.classes = ids
        else:
            class_check = np.in1d(self.classes, ids)
            if not np.all(class_check):
                msg = "Not all classes in the list are present in the directory. Missing classes: {}".format(
                    np.asarray(self.classes)[~class_check]
                )
                raise RuntimeError(msg)
            class_check = np.in1d(ids, self.classes)
            if not np.all(class_check):
                logging.basicConfig(format="%(message)s", level=logging.INFO)
                logging.info(
                    "Not all classes in the directory are present in the "
                    "classes list. Missing classes: %s. They will be ignored.",
                    (np.asarray(ids)[~class_check]),
                )

        paths = [
            datapath / p.name
            for p in paths
            for c in self.classes
            if c in p.name.split("_")[0]
        ]
        label_paths = (
            [
                label_path / p.name
                for p in label_paths
                for c in self.classes
                if c in p.name.split("_")[0]
            ]
            if label_path is not None
            else None
        )
        weight_paths = (
            [
                weight_path / p.name
                for p in weight_paths
                for c in self.classes
                if c in p.name.split("_")[0]
            ]
            if weight_path is not None
            else None
        )
        if self.dataset_size is not None:
            paths = paths[: self.dataset_size]

        if label_paths is not None and len(label_paths) != len(paths):
            msg = "Label paths and data paths do not match."
            raise RuntimeError(msg)
        if weight_paths is not None and len(weight_paths) != len(paths):
            msg = "Weight paths and data paths do not match."
            raise RuntimeError(msg)
        label_paths = label_paths if label_paths is not None else [None] * len(paths)
        weight_paths = weight_paths if weight_paths is not None else [None] * len(paths)

        raw_map_HDF5 = HDF5DataStore(datapath.joinpath("raw_map_data.h5"))
        label_HDF5 = (
            HDF5DataStore(label_path.joinpath("label_data.h5"))
            if label_paths is not None
            else None
        )
        weight_HDF5 = (
            HDF5DataStore(weight_path.joinpath("weight_data.h5"))
            if weight_paths is not None
            else None
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_dataset,
                    path,
                    label_path,
                    weight_path,
                    self.transformations,
                    self.augmentations,
                )
                for path, label_path, weight_path in zip(
                    paths, label_paths, weight_paths
                )
            ]

            for future in as_completed(futures):
                result = future.result()
                raw_map_HDF5.add_array(*result["map_data"])
                if result["label_data"] and label_HDF5:
                    label_HDF5.add_array(*result["label_data"])
                if result["weight_data"] and weight_HDF5:
                    weight_HDF5.add_array(*result["weight_data"])
                datasets.append(result)  # Collect processed datasets

        # Concat datasets if needed
        concatenated_data = [dataset["map_data"][0] for dataset in datasets]
        self.dataset = ConcatDataset(concatenated_data)

    def process(self, paths: list[str], datatype: str):
        """
        Process the loaded data with the specified transformations.

        Args:
            paths (list[str]): List of file paths to the data.
            datatype (str): Type of data being processed.

        Returns:
            DiskDataset: Processed dataset object.

        Raises:
            RuntimeError: If no transformations were provided.
        """

        raise NotImplementedError

    def get_loader(
        self,
        batch_size: int,
        split_size: float | None = None,
        no_val_drop: bool = False,
    ):
        """
        Retrieve the data loader.

        Args:
            batch_size (int): The batch size for the data loader.
            split_size (float | None, optional): The percentage of data to be used for validation set.
                If None, the entire dataset will be used for training. Defaults to None.
            no_val_drop (bool, optional): If True, the last batch of validation data will not be dropped if it is smaller than batch size. Defaults to False.

        Returns:
            DataLoader or Tuple[DataLoader, DataLoader]: The data loader(s) for testing or training/validation, according to whether training is True or False.

        Raises:
            RuntimeError: If split_size is None and the method is called for training.
            RuntimeError: If the train and validation sets are smaller than 2 samples.

        """
        if self.training:
            if split_size is None:
                msg = "Split size must be provided for training. "
                raise RuntimeError(msg)
            # split into train / val sets
            idx = np.random.permutation(len(self.dataset))
            if split_size < 1:
                split_size = split_size * 100

            s = int(np.ceil(len(self.dataset) * int(split_size) / 100))
            if s < 2:
                msg = "Train and validation sets must be larger than 1 sample, train: {}, val: {}.".format(
                    len(idx[:-s]), len(idx[-s:])
                )
                raise RuntimeError(msg)
            train_data = Subset(self.dataset, indices=idx[:-s])
            val_data = Subset(self.dataset, indices=idx[-s:])

            loader_train = DataLoader(
                train_data,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                drop_last=True,
            )
            loader_val = DataLoader(
                val_data,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                drop_last=(not no_val_drop),
            )
            return loader_train, loader_val

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )


class DiskDataset(AbstractDataset):
    """
    A dataset class for loading data from disk.

    Args:
        paths (list[str]): List of file paths.
        datatype (str, optional): Type of data to load. Defaults to "npy".
        rescale (int, optional): Rescale factor for the data. Defaults to 0.
        shiftmin (bool, optional): Whether to shift the minimum value of the data. Defaults to False.
        gaussianblur (bool, optional): Whether to apply Gaussian blur to the data. Defaults to False.
        normalise (bool, optional): Whether to normalise the data. Defaults to False.
        input_transform (typing.Any, optional): Additional input transformation. Defaults to None.
    """

    def __init__(
        self,
        paths: list[str],
        datatype: str = "npy",
        rescale: int = 0,
        shiftmin: bool = False,
        gaussianblur: bool = False,
        normalise: bool = False,
        input_transform: typing.Any = None,
    ) -> None:
        self.paths = paths
        self.rescale = rescale
        self.normalise = normalise
        self.gaussianblur = gaussianblur
        self.transform = input_transform
        self.datatype = datatype
        self.shiftmin = shiftmin

    def __len__(self):
        return len(self.paths)

    def dim(self):
        return len(np.array(self.read(self.paths[0])).shape)

    def __getitem__(self, item):
        filename = self.paths[item]

        data = np.array(self.read(filename))
        x = self.transformation(data)

        # ground truth
        y = Path(filename).name.split("_")[0]

        return x, y

    def read(self, filename):
        """
        Read data from file.

        Args:
            filename (str): File path.

        Returns:
            np.ndarray: Loaded data.

        Raises:
            RuntimeError: If the data type is not supported. Currently supported: .mrc, .npy
        """
        if self.datatype == "npy":
            return np.load(filename)

        if self.datatype == "mrc":
            try:
                with mrcfile.open(filename) as f:
                    return np.array(f.data)
            except ValueError as exc:
                msg = f"File {filename} is corrupted."
                raise ValueError(msg) from exc

        else:
            msg = "Currently we only support mrcfile and numpy arrays."
            raise RuntimeError(msg)

    def transformation(self, x):
        """
        Apply transformations to the input data.

        Args:
            x (np.ndarray): Input data.

        Returns:
            torch.Tensor: Transformed data.
        """
        if self.rescale:
            x = np.asarray(x, dtype=np.float32)
            sh = tuple([self.rescale / s for s in x.shape])
            x = zoom(x, sh)

        # convert numpy to torch tensor
        x = torch.Tensor(x)

        # unsqueeze adds a dimension for batch processing the data
        x = x.unsqueeze(0)

        if self.shiftmin:
            x = (x - x.min()) / (x.max() - x.min())

        if self.gaussianblur:
            T = transforms.GaussianBlur(3, sigma=(0.08, 10.0))
            x = T(x)

        if self.normalise:
            T = transforms.Normalize(0, 1, inplace=False)
            x = T(x)

        if self.transform:
            x = self.transform(x)
        return x

    def augment(self, augment):
        raise NotImplementedError


class MapDataset(AbstractDataset):
    """
    A dataset class for loading map data, alongside the corresponding class labels and weights.
    The map data is loaded from the disk and is decomposed into a set of tiles. These tiles are
    then reuturned when indexing the dataset.

    Args:

    Note: I'm not sure if shuffling will be used but the method I'm currently using will lazily
    load the data from disk so the map file will be loadeded, transformed and then the tile
    will be extracted. It might be good to include a cache option to store map data in memory.
    This could be useful to reduce the number of times the map data is loaded from disk...
    Perhaps saving them as hdf5 files would be a good idea?
    """

    def __init__(
        self,
        path: str | Path,
        label_path: str | Path | None = None,
        weight_path: str | Path | None = None,
        transforms: list[str] | None = None,
        augments: list[str] | None = None,
        decompose_kwargs: dict[str, int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.label_path = Path(label_path) if label_path is not None else None
        self.weight_path = Path(weight_path) if weight_path is not None else None
        self.mapobj = None
        self.label_mapobj = None
        self.weight_mapobj = None
        self.slices = None
        self.tiles = None
        self.tiles_count: int = 0
        self.transforms = transforms
        self.augments = augments
        self.transform_kwargs = None
        if decompose_kwargs is None:
            decompose_kwargs = {"cshape": 64, "margin": 8}

        if self.transform_kwargs is None:
            self.transform_kwargs = {}

        if not decompose_kwargs.get("step", False):
            decompose_kwargs["step"] = decompose_kwargs.get("cshape", 1) - (
                2 * decompose_kwargs.get("margin")
            )

        self.decompose_kwargs = decompose_kwargs

    def __len__(self):
        # TODO: The tile counts need to be calculated before __getitem__ is called
        # The amount of tiles is linked to the transformations applied to the map data
        # This would mean the best place to calculate the tile count would be in the __init__
        # method and subsequently the transform method would need to be called there too

        # 1 represents the full map
        return self.tiles_count if self.tiles_count != 0 else 1

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # start by loading the map data
        self.load_map_objects()

        self.transform()
        _ = self.augment()
        # SEND TO HDF5 FILE to be saved, some will be duplicates so need to keep track of the duplicates

        if (self.slices is None) or (self.tiles is None):
            decompose = DecomposeToSlices(
                self.mapobj,
                step=self.decompose_kwargs.get("step"),
                cshape=self.decompose_kwargs.get("cshape"),
                margin=self.decompose_kwargs.get("margin"),
            )  # TODO: move this
            self.slices = decompose.slices
            self.tiles = decompose.tiles
            self.tiles_count = len(self.tiles)

        map_slice = self.mapobj.data[self.slices[idx]]
        label_slice = (
            self.label_mapobj.data[self.slices[idx]]
            if self.label_mapobj is not None
            else None
        )
        weight_slice = (
            self.weight_mapobj.data[self.slices[idx]]
            if self.weight_mapobj is not None
            else None
        )

        # Close the map objects
        self.close_map_objects()

        return (
            torch.tensor(map_slice),
            torch.tensor(label_slice) if label_slice is not None else None,
            torch.tensor(weight_slice) if weight_slice is not None else None,
        )

    def _transform_keywords_builder(self):
        keywords = {}
        keywords.update(self.decompose_kwargs)

        for transform in self.transforms:
            if transform == Transforms.MASKCROP.value:
                keywords["mask"] = self.label_mapobj
            if transform == Transforms.NORM.value:
                keywords["ext_dim"] = (0, 0, 0)
                keywords["fill_padding"] = (0, 0, 0)
            if transform == Transforms.VOXNORM.value:
                keywords["vox"] = self.decompose_kwargs.get("vox", 1.0)
                keywords["vox_lim"] = self.decompose_kwargs.get("vox_lim", (0.95, 1.05))

        return keywords

    def _augment_keywords_builder(self):
        keywords = {}
        for augment in self.augments:
            if augment.__class__.__name__ == "RandomRotationAugment":
                keywords["ax"] = self.ax
                keywords["an"] = self.an

        return keywords

    def load_map_objects(
        self,
    ) -> None:
        self.mapobj = get_mapobjhandle(self.path)
        if self.label_path is not None:
            if not self.label_path.exists():
                msg = f"Label file {self.label_path} not found."
                raise FileNotFoundError(msg)
            self.label_mapobj = get_mapobjhandle(self.label_path)
        if self.weight_path is not None:
            if not self.weight_path.exists():
                msg = f"Weight file {self.weight_path} not found."
                raise FileNotFoundError(msg)
            self.weight_mapobj = get_mapobjhandle(self.weight_path)

    def close_map_objects(self, *args):
        for arg in args:
            if arg is not None:
                arg.close()

    def augment(self) -> None:
        augment_kwargs = self._augment_keywords_builder()
        augment_kwargs["retall"] = True
        if len(self.augments) == 0:
            return {}

        self.mapobj, extra_kwargs = ComposeAugment(self.augments)(
            self.mapobj, **augment_kwargs
        )
        augment_kwargs["retall"] = False
        augment_kwargs.update(
            extra_kwargs
        )  # update the kwargs with the returned values

        self.label_mapobj = ComposeAugment(self.augments)(
            self.label_mapobj, **augment_kwargs
        )
        self.weight_mapobj = ComposeAugment(self.augments)(
            self.weight_mapobj, **augment_kwargs
        )

        return augment_kwargs

    def transform(self):
        # TODO: Need to see if same transforms are applied to all map objects, maybe just voxel space normalisation
        transform_kwargs = self._transform_keywords_builder()
        if len(self.transforms) == 0:
            self.transform_kwargs = transform_kwargs

        self.transform_kwargs = ComposeTransform(self.transforms)(
            self.mapobj, **transform_kwargs
        )


def process_dataset(path, label_path, weight_path, transformations, augmentations):
    map_dataset = MapDataset(
        path,
        label_path=label_path,
        weight_path=weight_path,
        transforms=transformations,
        augments=augmentations,
    )
    map_dataset.load_map_objects()
    map_dataset.transform()
    map_dataset.augment()
    result = {
        "map_data": (map_dataset.mapobj.data, f"{path.stem}_map"),
        "label_data": (map_dataset.label_mapobj.data, f"{path.stem}_label")
        if label_path is not None
        else None,
        "weight_data": (map_dataset.weight_mapobj.data, f"{path.stem}_weight")
        if weight_path is not None
        else None,
    }
    map_dataset.close_map_objects()
    return result
