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

# import numpy as np
# from utils import nearest_neighbors_2
#
# voxel_points = np.array([[0, 0, 0], [1, 0, 1], [2, 1, 0], [3, 1, 1], [4, 0, 0], [5, 0, 1], [6, 1, 0], [7, 1, 1]])
# voxel_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# transformed_scan_points = np.array(
#     [[4.1, 0.8, 0.1], [3.1, 0.1, 0.9], [7.1, 0.9, 0.1], [1.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.9, 0.1, 0.9],
#      [0.9, 0.6, 0.1], [0.3, 0.2, 0.9]])
#
# voxel_mask = np.zeros(voxel_points.shape[0], dtype=bool)
#
# dists, voxel_indices = nearest_neighbors_2(voxel_points, transformed_scan_points, k_nn=1)
# labels = voxel_labels[voxel_indices].astype(np.uint8)
# voxel_mask[voxel_indices] = True
#
# print(voxel_indices)
#
# voxel_points = voxel_points[voxel_mask]
#
# # create a mask that maps old indices to new indices
# new_map = np.cumsum(voxel_mask) - 1
#
# voxel_map = new_map[voxel_indices]
#
# print(voxel_points)
# print(voxel_map)

def create_window_ranges(scans: list, window_size: int = 200):
    """ Create a list of tuples containing the start and end indices of the windows.
    Example with window_size = 200 and 1100 scans:
    [(0, 199), (200, 399), (400, 599), (600, 799), (800, 999), (1000, 1099)]
    """
    window_ranges = []
    for i in range(0, len(scans), window_size):
        window_ranges.append((i, min(i + window_size - 1, len(scans) - 1)))
    return window_ranges


scans = [i for i in range(1100)]
window_ranges = create_window_ranges(scans, window_size=200)
print(window_ranges)
