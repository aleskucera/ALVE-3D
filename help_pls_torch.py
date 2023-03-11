#!/usr/bin/env python
import time

import torch
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
            cloud = VoxelCloud(size=4, device=torch.device('gpu'))
            cloud.add_values(values=torch.tensor([1, 2, 3, 4, 5, 6]), voxel_map=torch.tensor([0, 0, 0, 1, 1, 1]))
            cloud.add_values(values=torch.tensor([2, 1]), voxel_map=torch.tensor([0, 3]))
            stds = cloud.std()
            print(stds)

        The output should be:
            tensor([..., ..., nan, 0.], device='cuda:0')

        There is also an active voxels tensor, which stores whether a voxel is active or not. If a voxel is active, it
        means that it has been already labeled, and it is not necessary to label it again, so it can be ignored.
        For example if in example above we had added the same values, but the active voxels would be:
            - Voxel 0: active
            - Voxel 1: inactive
            - Voxel 2: inactive
            - Voxel 3: inactive

        Then the output should be:
            tensor([nan, ..., nan, 0.], device='cuda:0')
        """

    def __init__(self, size: int, device: torch.device):
        self.size = size
        self.device = device

        # Values are stored in a 1D tensor, when new values are added, they are concatenated to the existing values
        self.values = torch.zeros((0,), dtype=torch.float32, device=device)

        # Voxel map is a 1D tensor that maps each value to a voxel
        self.voxel_map = torch.zeros((0,), dtype=torch.int32, device=device)

        # Active voxels is a 1D tensor that stores whether a voxel is active or not. If a voxel is active, it means that
        # it has been already labeled, and it is not necessary to label it again, so it can be ignored.
        self.active_voxels = torch.zeros(size, dtype=torch.bool, device=device)

    def add_values(self, values: torch.Tensor, voxel_map: torch.Tensor):
        """ Function which adds new values to the cloud."""

        # Calculate which voxels are inactive and should be added to the cloud for calculation
        inactive_voxels = torch.where(self.active_voxels == 0)[0]

        # Get indices of inactive voxels in the new values
        indices = torch.where(torch.isin(voxel_map, inactive_voxels))

        # Remove active voxels from the new values
        inactive_values = values[indices]
        inactive_voxel_map = voxel_map[indices]

        # concatenate new values to existing values
        self.values = torch.cat((self.values, inactive_values), dim=0)
        self.voxel_map = torch.cat((self.voxel_map, inactive_voxel_map), dim=0)

    def get_std_values(self):
        values = torch.full((self.size,), float('nan'), dtype=torch.float32, device=self.device)

        # Sort the values ascending by voxel map
        sorted_indices = torch.argsort(self.voxel_map)
        sorted_values = self.values[sorted_indices]

        # Split the values to a multidimensional tensor, where each row contains the values of a voxel
        unique_voxels, counts = torch.unique(self.voxel_map, return_counts=True)
        voxel_values = torch.split(sorted_values, counts.tolist())

        # Calculate the standard deviation of each voxel and assign it to the corresponding voxel
        voxel_stds = []
        for v in tqdm(voxel_values):
            voxel_stds.append(torch.std(v))

        voxel_stds = torch.tensor(voxel_stds, device=self.device)
        # voxel_stds = torch.tensor([torch.std(v) for v in voxel_values])
        values[unique_voxels] = voxel_stds

        return values


if __name__ == '__main__':
    print(VoxelCloud.__doc__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example usage
    print('\nExample usage:\n')
    size = 10
    values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32, device=device)
    voxel_map = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long, device=device)

    cloud = VoxelCloud(size, device)
    cloud.add_values(values, voxel_map)

    print(f'\t - Values: {cloud.values}')
    print(f'\t - Voxel map: {cloud.voxel_map}')

    std_values = cloud.get_std_values()

    print(f'\t - Standard deviation of each voxel: {std_values}\n')

    time.sleep(1)

    # What is already problem:
    print('\nWhat is already a problem (dataset is much bigger):\n')
    size = 1000000
    values = torch.rand(500000)
    voxel_map = torch.randint(0, size, (500000,), dtype=torch.long)

    cloud = VoxelCloud(size, torch.device('cpu'))
    cloud.add_values(values, voxel_map)

    start = time.time()
    std_values = cloud.get_std_values()
    end = time.time()
    print(f'\t - Time: {end - start}\n')
