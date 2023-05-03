import time
import torch

from torch_scatter import scatter_mean


def test_scatter_mean():
    # Values are tensor of shape (N, C) where N is the number of voxels and C is the number of classes.
    # Voxel map is a tensor of shape (N,) where N is the number of voxels.
    values = torch.rand((10000000, 10))
    voxel_map = torch.randint(0, 200000, (10000000,))

    start = time.time()
    metric_normal = torch.full((200000, 10), float('nan'), dtype=torch.float32)
    order = torch.argsort(voxel_map)
    unique_voxels, num_views = torch.unique(voxel_map, return_counts=True)

    values2 = values[order]
    mapping = unique_voxels.type(torch.long)
    value_sets = torch.split(values2, num_views.tolist())

    vals = torch.tensor([])
    for value_set in value_sets:
        vals = torch.cat((vals, torch.mean(value_set, dim=0).unsqueeze(0)), dim=0)

    metric_normal[mapping] = vals

    print(f'Calculating metric for took {time.time() - start} seconds.')

    start = time.time()
    metric_scatter = torch.full((1000,), float('nan'), dtype=torch.float32)
    metric_scatter[mapping] = scatter_mean(values, voxel_map, dim=0)
    print(f'Calculating metric for took {time.time() - start} seconds.')

    print(torch.allclose(metric_normal, metric_scatter))


def test_scatter_mean2():
    src = torch.tensor([[2, 0, 1, 4, 3],
                        [0, 1, 1, 3, 4],
                        [0, 2, 1, 3, 4]])
    index = torch.tensor([4, 5, 4])

    out = scatter_mean(src, index, dim=0, dim_size=8)
    print(out)


def main():
    test_scatter_mean2()
    # size = 4
    # tensor = torch.tensor([1, 2, 3, 4])
    # full_tensor = torch.concatenate((tensor, torch.zeros((size - tensor.shape[0],))))
    # print(full_tensor)


if __name__ == '__main__':
    main()
