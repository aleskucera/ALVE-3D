# import torch
#
# # Create tensor of shape (2, 4, 3, 3)
# proj_scan = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
#                           [[[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                            [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]])
# print(proj_scan.shape)
#
# # Create tensor of shape (2, 3, 3)
# proj_voxel_map = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
# print(proj_voxel_map.shape)
#
# # Transpose proj_scan to shape (2, 3, 3, 4)
# proj_scan = proj_scan.permute(0, 2, 3, 1)
# print(proj_scan.shape)
#
# # Create from proj_scan a tensor of shape (2x3x3, 4)
# proj_scan = proj_scan.reshape(-1, proj_scan.shape[-1])
# print(proj_scan.shape)
#
# # Create from proj_voxel_map a tensor of shape (2x3x3)
# proj_voxel_map = proj_voxel_map.reshape(-1)
# print(proj_voxel_map.shape)
#
# print(proj_scan)
# print(proj_voxel_map)
import torch
import os
import numpy as np


def find_index(arr: np.ndarray, number: int) -> int:
    sizes = np.zeros_like(arr, dtype=np.int32)
    for i, path in enumerate(arr):
        file_name = os.path.basename(path)
        bounds = file_name.split('.')[0].split('_')
        size = int(bounds[1]) - int(bounds[0]) + 1
        sizes[i] = size

    cum_sizes = np.cumsum(sizes)
    index = np.where(cum_sizes >= number)[0][0]
    return index


arr = np.array(['/path/to/the/file/000100_000199.h5',
                '/path/to/the/file/000200_000299.h5',
                '/path/to/the/file/000300_000399.h5'])

index = find_index(arr, 250)  # Returns 0
print(index)
