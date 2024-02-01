from __future__ import annotations

import logging
import os
import random
import typing

import mrcfile
import numpy as np
import torch
from scipy.ndimage import zoom
from torchvision import transforms

from .base import AbstractDataLoader, AbstractDataset


class DiskDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_size: int | None = None,
        save_to_disk: bool = False,
        training: bool = True,
        classes: list[str] | None = None,
        pipeline: str = "disk",
    ) -> None:
        if classes is None:
            classes = []
        super().__init__(pipeline, classes, dataset_size, save_to_disk, training)

    def load(self, datapath, datatype):
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
                    np.asarray(ids)[~class_check]
                )
                raise RuntimeError(msg)
            class_check = np.in1d(ids, self.classes)
            if not np.all(class_check):
                logging.info(
                    "Not all classes in the directory are present in the "
                    "classes list. Missing classes: {}. They will be ignored.".format(
                        np.asarray(ids)[~class_check]
                    )
                )

            # subset affinity matrix with only the relevant classes

        paths = [p for p in paths for c in self.classes if c in p.split("_")[0]]
        if self.dataset_size is not None:
            paths = paths[: self.dataset_size]

        self.dataset = DiskDataset(paths=paths, datatype=datatype)
        return super().load()

    def process(self):
        return super().process()

    def get_loader(self, split_size: float, batch_size: int):
        return super().get_loader(split_size, batch_size)


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
        self.rescale = rescale
        self.transform = input_transform
        self.datatype = datatype
        self.shiftmin = shiftmin
        super().__init__()

    def __len__(self):
        return len(self.paths)

    def dim(self):
        return len(np.array(self.read(self.paths[0])).shape)

    def __getitem__(self, item):
        filename = self.paths[item]

        data = np.array(self.read(filename))
        x = self.transformation(data)

        # ground truth
        y = os.path.basename(filename).split("_")[0]

        return x, y

    def read(self, filename):
        if self.datatype == "npy":
            return np.load(filename)

        elif self.datatype == "mrc":
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
