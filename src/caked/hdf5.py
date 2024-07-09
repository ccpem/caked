from __future__ import annotations

import h5py
import numpy as np


class HDF5DataStore:
    def __init__(self, save_path: str):
        self.save_path = save_path

    def __getitem__(self, key: str):
        with h5py.File(self.save_path, "r") as f:
            return np.array(f[key])

    def get(self, key: str, default=None):
        try:
            with h5py.File(self.save_path, "r") as f:
                return np.array(f[key])
        except KeyError:
            return default

    def __len__(self):
        return len(self.keys())

    def add_array(
        self, array: np.ndarray, dataset_name: str, compression: str = "gzip"
    ) -> str:
        if self.check_name_in_store(dataset_name):
            dataset_name = self._add_number_to_dataset_name(dataset_name)
        with h5py.File(self.save_path, "a") as f:  # Open in append mode
            f.create_dataset(dataset_name, data=array, compression=compression)

        return dataset_name

    def save(self, array_list: list[np.ndarray]):
        for i, array in enumerate(array_list):
            self.add_array(array, f"array_{i}")

    def check_name_in_store(self, dataset_name: str):
        if not self.save_path.exists():
            return False
        with h5py.File(self.save_path, "r") as f:
            return dataset_name in f

    def _add_number_to_dataset_name(self, dataset_name: str, delimiter: str = "--"):
        # add a number to the end of the dataset name, take the last number and increment it
        existing_names = [name for name in self.keys() if dataset_name in name]
        last_number = (
            max(
                [
                    int(name.split(delimiter)[0])
                    for name in existing_names
                    if delimiter in name
                ]
            )
            if len(existing_names) > 1
            else 0
        )

        # dataset_name = dataset_name.split(delimiter)[0:-1]
        return f"{last_number+1}{delimiter}{dataset_name}"

    def keys(self):
        with h5py.File(self.save_path, "r") as f:
            return list(f.keys())
