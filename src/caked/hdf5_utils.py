# create a class to handle hdf5 files
import h5py
import numpy as np

# Want to take an input of np arrays and store them in a hdf5 file

# also


class HDF5DataStore:
    def __init__(self, save_path):
        self.save_path = save_path

    def add_array(self, array, dataset_name, compression="gzip"):
        with h5py.File(self.save_path, "a") as f:  # Open in append mode
            f.create_dataset(dataset_name, data=array, compression=compression)
        print(f"Dataset {dataset_name} added to {self.save_path}")

    def save(self, array_list):
        for i, array in enumerate(array_list):
            self.add_array(array, f"array_{i}")


# Assuming raw_map_HDF5, label_HDF5, and weight_HDF5 are instances of BatchHDF5Writer
# Initialize them with the HDF5 file and desired batch size

# load in map dataset, perform the transformations inside the dataset init


# first test of loading the files one by one took 13 minutes 10 seconds
