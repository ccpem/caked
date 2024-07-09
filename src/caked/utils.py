from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset

from caked.hdf5 import HDF5DataStore
from caked.Wrappers import none_return_none


def process_datasets(
    num_workers: int,
    paths: list[str],
    label_paths: list[str],
    weight_paths: list[str],
    transformations,
    augmentations,
    decompose: bool,
    raw_map_HDF5: HDF5DataStore,
    label_HDF5: HDF5DataStore | None = None,
    weight_HDF5: HDF5DataStore | None = None,
):
    """
    Process multiple datasets in parallel.

    Args:
        num_workers: Number of workers to use.
        paths: List of paths to the map files.
        label_paths: List of paths to the label files.
        weight_paths: List of paths to the weight files.
        raw_map_HDF5: Instance of HDF5DataStore to store map data.
        label_HDF5: Instance of HDF5DataStore to store label data.
        weight_HDF5: Instance of HDF5DataStore to store weight data.

    Returns:
        None

    """
    datasets = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_map_dataset,
                path,
                label_path,
                weight_path,
                transformations,
                augmentations,
                decompose,
                raw_map_HDF5,
                label_HDF5,
                weight_HDF5,
            )
            for path, label_path, weight_path in zip(paths, label_paths, weight_paths)
        ]

        for future in as_completed(futures):
            result, dataset = future.result()

            add_dataset_to_HDF5(
                *result.values(),
                dataset.id,
                raw_map_HDF5,
                label_HDF5=label_HDF5,
                weight_HDF5=weight_HDF5,
            )
            datasets.append(dataset)  # Collect processed datasets

    return datasets


def process_map_dataset(
    path: str | Path,
    label_path: str | Path | None,
    weight_path: str | Path | None,
    transformations: list[str],
    augmentations: list[str],
    decompose: bool,
    map_hdf5: HDF5DataStore,
    label_hdf5: HDF5DataStore | None,
    weight_hdf5: HDF5DataStore | None,
):
    """
    Process a single map dataset, applying transformations and augmentations, closes the map objects.

    Args:
        path: (str| Path) path to the map file.
        label_path: (str | Path | None) path to the label file.
        weight_path: (str | Path | None) path to the weight file.
        transformations: (list[str]) list of transformations to apply.
        augmentations: (list[str]) list of augmentations to apply.

    Returns:
        tuple[dict, MapDataset]: dictionary containing map, label, and weight data,
        and the processed MapDataset object.


    """
    from caked.dataloader import MapDataset  # Avoid circular import

    map_dataset = MapDataset(
        path,
        label_path=label_path,
        weight_path=weight_path,
        transforms=transformations,
        augments=augmentations,
        decompose=decompose,
        map_hdf5_store=map_hdf5,
        label_hdf5_store=label_hdf5,
        weight_hdf5_store=weight_hdf5,
    )
    map_dataset.transform(close_map_objects=False)
    map_dataset.augment(close_map_objects=False)
    result = {
        "map_data": map_dataset.mapobj.data,
        "label_data": map_dataset.label_mapobj.data if label_path is not None else None,
        "weight_data": map_dataset.weight_mapobj.data
        if weight_path is not None
        else None,
    }

    map_dataset.close_map_objects()

    return result, map_dataset


def add_dataset_to_HDF5(
    map_data: np.ndarray,
    label_data: np.ndarray | None,
    weight_data: np.ndarray | None,
    name: str,
    raw_map_HDF5: HDF5DataStore,
    label_HDF5: HDF5DataStore | None = None,
    weight_HDF5: HDF5DataStore | None = None,
) -> tuple[str, str, str]:
    """
    Add a map data to HDF5 files.

    Args:

        map_data: (np.ndarray) map data
        raw_map_HDF5: (HDF5DataStore) instance of HDF5DataStore to store map data
        name: (str) name of the dataset
        label_data: (np.ndarray | None) label data
        weight_data: (np.ndarray | None) weight data
        label_HDF5: (HDF5DataStore | None) instance of HDF5DataStore to store label data
        weight_HDF5: (HDF5DataStore | None) instance of HDF5DataStore to store weight data

    Returns:
        tuple[str, str, str]: map_id, label_id, weight_id
    """
    map_id = f"{name}_map"
    label_id = f"{name}_label"
    weight_id = f"{name}_weight"

    map_id = raw_map_HDF5.add_array(map_data, map_id)
    if label_HDF5 is not None:
        label_id = label_HDF5.add_array(label_data, label_id)
    if weight_HDF5 is not None:
        weight_id = weight_HDF5.add_array(weight_data, weight_id)

    return map_id, label_id, weight_id


# Functions so I don't need to write out if xxx is None each time


@none_return_none
def filter_and_construct_paths(base_path, paths, classes):
    return [
        base_path / p.name for p in paths for c in classes if c in p.name.split("_")[0]
    ]


def duplicate_and_augment_from_hdf5(
    map_data_loader,
    ids: list[str],
    augmentations: list[str] | None = None,
):
    """
    Add data from a list of paths to the HDF5 store.

    Args:
        pathnames (list[str]): List of path names accessed from the HDF5 store, typically the stem of the original file.

    Returns:
        None
    """
    from caked.dataloader import ArrayDataset, MapDataLoader

    datasets = map_data_loader.dataset.datasets

    if not isinstance(map_data_loader, MapDataLoader):
        msg = "map_data_loader must be an instance of MapDataLoader."
        raise TypeError(msg)

    if len(map_data_loader.dataset.datasets) == 0:
        msg = "No datasets have been loaded yet."
        raise RuntimeError(msg)

    map_hdf5_store, label_hdf5_store, weight_hdf5_store = (
        map_data_loader.dataset.datasets[0].map_hdf5_store,
        map_data_loader.dataset.datasets[0].label_hdf5_store,
        map_data_loader.dataset.datasets[0].weight_hdf5_store,
    )

    for dataset_id in ids:
        array = map_hdf5_store[dataset_id + "_map"]
        label_array = (
            label_hdf5_store.get(dataset_id + "_label")
            if label_hdf5_store is not None
            else None
        )
        weight_array = (
            weight_hdf5_store.get(dataset_id + "_weight")
            if weight_hdf5_store is not None
            else None
        )

        dataset = ArrayDataset(
            dataset_id=dataset_id,
            data_array=array,
            label_array=label_array,
            weight_array=weight_array,
            augments=augmentations,
            map_hdf5_store=map_hdf5_store,
            label_hdf5_store=label_hdf5_store,
            weight_hdf5_store=weight_hdf5_store,
            decompose=map_data_loader.dataset.datasets[0].decompose,
        )

        dataset.augment()
        dataset.save_to_store()

        datasets.append(dataset)

    map_data_loader.dataset = ConcatDataset(datasets)
