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
import matplotlib.pyplot as plt


class MyDataProcessor:

    @staticmethod
    def _interpolate_values(values: torch.Tensor, min_selected_value: float) -> torch.Tensor:
        # sort values in descending order
        sorted_indices = torch.argsort(values, descending=True)
        sorted_values = values[sorted_indices]

        # interpolate the values using a linear interpolation function
        f = torch.nn.functional.interpolate
        expected_size = int(0.01 * sorted_values.shape[0])
        interp_values = f(sorted_values.unsqueeze(0).unsqueeze(0), size=expected_size, mode='linear',
                          align_corners=True).squeeze()
        # identify the threshold index
        threshold_index = torch.nonzero(interp_values < min_selected_value, as_tuple=True)[0]

        # create the downsampled x-axis values
        interp_values = interp_values.numpy()
        x = torch.linspace(0, 1, len(interp_values))

        # plot the original and interpolated values
        plt.plot(x[:threshold_index], interp_values[:threshold_index], 'b', label='larger than threshold')
        plt.plot(x[threshold_index:], interp_values[threshold_index:], 'r', label='smaller than threshold')
        plt.axhline(y=0.5, color='gray', linestyle='--')
        plt.legend()
        plt.show()

        return interp_values


# create a random tensor of 10000 elements
values = torch.randn(10000)

# set the minimum selected value threshold
min_selected_value = 0.5

# downsample the tensor using interpolation and plot the resulting graph
downsampled_values = MyDataProcessor._interpolate_values(values, min_selected_value)

# print the shape of the downsampled tensor
print("Downsampled tensor shape:", downsampled_values.shape)
