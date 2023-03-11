#!/usr/bin/env python
import time

import numpy as np
from tqdm import tqdm


class VoxelCloud(object):
    """ This class represents a voxel cloud. This can be very confusing, but the main things are:

    - The size is number of voxels in the cloud
    - After initializing the cloud, we can append values to the voxels. This can be done multiple times, by calling the
    add_values function. The values are stored in a 1D tensor, and the voxel_map is a 1D tensor that maps each value to a
    voxel.
    - After adding multiple values, we can calculate for example standard deviation of the values in each voxel.

    Example:
        We create voxel cloud of size 4. (4 voxels)
        Then we add values [1, 2, 3] to voxel 0, and values [4, 5, 6] to voxel 1.
        Then we add values [2] to voxel 0, and [1] to voxel 3.
        After that we can calculate the standard deviation of the values in each voxel e.g.:
            - Voxel 0: std([1, 2, 3, 2]) = ...
            - Voxel 1: std([4, 5, 6]) = ...
            - Voxel 2: std([]) = nan
            - Voxel 3: std([1]) = 0.

        The code for this should look something like this:
            cloud = VoxelCloud(size=4)
            cloud.add_values(values=np.ndarray([1, 2, 3, 4, 5, 6]), voxel_map=np.ndarray([0, 0, 0, 1, 1, 1]))
            cloud.add_values(values=np.ndarray([2, 1]), voxel_map=np.ndarray([0, 3]))
            stds = cloud.std()
            print(stds)

        The output should be:
            np.ndarray([..., ..., nan, 0.])

        There is also an active voxels array, which stores whether a voxel is active or not. If a voxel is active, it
        means that it has been already labeled, and it is not necessary to label it again, so it can be ignored.
        For example if in example above we had added the same values, but the active voxels would be:
            - Voxel 0: active
            - Voxel 1: inactive
            - Voxel 2: inactive
            - Voxel 3: inactive

        Then the output should be:
            np.ndarray([nan, ..., nan, 0.])
        """

    def __init__(self, size: int):
        self.size = size

        # Values are stored in a 1D tensor, when new values are added, they are concatenated to the existing values
        self.values = np.zeros((0,), dtype=np.float32)

        # Voxel map is a 1D tensor that maps each value to a voxel
        self.voxel_map = np.zeros((0,), dtype=np.int32)

        # Active voxels is a 1D tensor that stores whether a voxel is active or not. If a voxel is active, it means that
        # it has been already labeled, and it is not necessary to label it again, so it can be ignored.
        self.active_voxels = np.zeros(size, dtype=bool)

    def add_values(self, values: np.ndarray, voxel_map: np.ndarray):
        """ Function which adds new values to the cloud."""

        # Calculate which voxels are inactive and should be added to the cloud for calculation
        inactive_voxels = np.where(self.active_voxels == 0)[0]

        # Get indices of inactive voxels in the new values
        indices = np.where(np.isin(voxel_map, inactive_voxels))

        # Remove active voxels from the new values
        inactive_values = values[indices]
        inactive_voxel_map = voxel_map[indices]

        # concatenate new values to existing values
        self.values = np.concatenate((self.values, inactive_values))
        self.voxel_map = np.concatenate((self.voxel_map, inactive_voxel_map))

    def get_std_values(self):
        """ Function which calculates the standard deviation of each voxel. Standard deviation is only an example, any
        other function can be used."""

        # Create an array to store the values, and initialize it with NaN
        values = np.full((self.size,), float('nan'), dtype=np.float32)

        # Iterate over all voxels and calculate the standard deviation of the values in each voxel
        # TODO: This for cycle is the problem, it takes too long
        for voxel in tqdm(np.unique(self.voxel_map), desc='Calculating standard deviation'):
            voxel_values = self.values[self.voxel_map == voxel]
            voxel_std = np.std(voxel_values)
            values[voxel] = voxel_std
        return values


if __name__ == '__main__':
    print(VoxelCloud.__doc__)

    # Example usage
    print('\nExample usage:\n')
    size = 10
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    voxel_map = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    cloud = VoxelCloud(size)
    cloud.add_values(values, voxel_map)

    print(f'\t - Values: {cloud.values}')
    print(f'\t - Voxel map: {cloud.voxel_map}')

    std_values = cloud.get_std_values()

    print(f'\t - Standard deviation of each voxel: {std_values}\n')

    time.sleep(1)

    # What is already problem:
    print('\nWhat is already a problem (and dataset is much bigger):\n')
    size = 1000000
    values = np.random.rand(500000)
    voxel_map = np.random.randint(0, size, 500000)

    cloud = VoxelCloud(size)
    cloud.add_values(values, voxel_map)

    start = time.time()
    std_values = cloud.get_std_values()
    end = time.time()
    print(f'\t - Time: {end - start}\n')
