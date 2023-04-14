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
import numpy as np
import h5py
import matplotlib.pyplot as plt

with h5py.File('experiment.h5', 'w') as f:
    # Groups
    f.create_group('selection_mask')
    f.create_group('label_mask')

    # Datasets
    f.create_dataset('selection_mask/scans', data=np.zeros((10,), dtype=bool))
    f.create_dataset('selection_mask/clouds', data=np.ones((3,), dtype=bool))
    f.create_dataset('label_mask/scans', data=np.zeros((10,), dtype=bool))
    f.create_dataset('label_mask/clouds', data=np.ones((3,), dtype=bool))

# Open the file
with h5py.File('experiment.h5', 'r') as f:
    scans_selection_mask = np.asarray(f['selection_mask']['scans'])
    clouds_selection_mask = np.asarray(f['selection_mask']['clouds'])
    scans_label_mask = np.asarray(f['label_mask']['scans'])
    clouds_label_mask = np.asarray(f['label_mask']['clouds'])

print(scans_selection_mask)
print(clouds_selection_mask)
print(scans_label_mask)
print(clouds_label_mask)
