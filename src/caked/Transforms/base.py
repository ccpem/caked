from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from ccpem_utils.map.parse_mrcmapobj import MapObjHandle


class TransformBase(ABC):
    """
    Base class for transformations.

    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data):
        msg = "The __call__ method must be implemented in the subclass"
        raise NotImplementedError(msg)


class AugmentBase(ABC):
    """
    Base class for augmentations.
    """

    # This will need to take the hyper parameters for the augmentations

    @abstractmethod
    def __init__(self, random_seed: int = 42):
        self.random_state = np.random.RandomState(random_seed)

    @abstractmethod
    def __call__(self, data, **kwargs):
        msg = "The __call__ method must be implemented in the subclass"
        raise NotImplementedError(msg)


class MapObjTransformBase(TransformBase):
    """
    Base class for transformations that operate on MapObjHandle objects.

    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    def __call__(self, mapobj: MapObjHandle, **kwargs) -> MapObjHandle:
        if not isinstance(mapobj, MapObjHandle):
            msg = "mapobj must be an instance of MapObjHandle"
            raise TypeError(msg)
        # Proceed with the method implementation after the check
        msg = "The __call__ method must be implemented in the subclass"
        raise NotImplementedError(msg)
