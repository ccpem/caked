from __future__ import annotations

import logging
import os
import random
import typing
from pathlib import Path

import mrcfile
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .base import AbstractDataLoader, AbstractDataset

np.random.seed(42)
TRANSFORM_OPTIONS = ["rescale", "normalise", "gaussianblur", "shiftmin"]


class DiskDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_size: int | None = None,
        save_to_disk: bool = False,
        training: bool = True,
        classes: list[str] | None = None,
        pipeline: str = "disk",
        transformations: str | None = None,
    ) -> None:
        self.dataset_size = dataset_size
        self.save_to_disk = save_to_disk
        self.training = training
        self.pipeline = pipeline
        self.transformations = transformations
        if classes is None:
            self.classes = []
        else:
            self.classes = classes

    def load(self, datapath, datatype) -> None:
        paths = [f for f in os.listdir(datapath) if "." + datatype in f]

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
        if self.transformations is None:
            msg = "No processing to do as no transformations were provided."
            raise RuntimeError(msg)
        transforms = self.transformations.split(",")
        rescale, normalise, gaussianblur, shiftmin = np.in1d(
            TRANSFORM_OPTIONS, transforms
        )
        return DiskDataset(
            paths=paths,
            datatype=datatype,
            rescale=rescale,
            normalise=normalise,
            gaussianblur=gaussianblur,
            shiftmin=shiftmin,
        )

    def get_loader(self, batch_size: int, split_size: float | None = None):
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
            )
            loader_val = DataLoader(
                val_data,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
            )
            return loader_val, loader_train

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )


class DiskDataset(AbstractDataset):
    def __init__(
        self,
        paths: list[str],
        datatype: str = "npy",
        rescale: bool = False,
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
        if self.datatype == "npy":
            return np.load(filename)

        if self.datatype == "mrc":
            with mrcfile.open(filename) as f:
                return np.array(f.data)

        else:
            msg = "Currently we only support mrcfile and numpy arrays."
            raise RuntimeError(msg)

    def transformation(self, x):
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
