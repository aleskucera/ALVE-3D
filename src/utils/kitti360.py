import os

import torch
import numpy as np

from src.models.pointnet_sp import STNkD, PointNet


def create_model(args):
    """ Creates model """
    model = torch.nn.Module()
    if args.learned_embeddings and 'ptn' in args.ptn_embedding and args.ptn_nfeat_stn > 0:
        model.stn = STNkD(args.ptn_nfeat_stn, args.ptn_widths_stn[0], args.ptn_widths_stn[1],
                          norm=args.ptn_norm,
                          n_group=args.ptn_n_group)

    if args.learned_embeddings and 'ptn' in args.ptn_embedding:
        n_embed = args.ptn_widths[1][-1]
        n_feat = 3 + 3 * args.use_rgb
        nfeats_global = len(args.global_feat) + 4 * args.stn_as_global + 1  # we always add the diameter
        model.ptn = PointNet(args.ptn_widths[0], args.ptn_widths[1], [], [], n_feat, 0,
                             prelast_do=args.ptn_prelast_do,
                             nfeat_global=nfeats_global, norm=args.ptn_norm, is_res=False,
                             last_bn=True)  # = args.normalize_intermediary==0)

    if args.ver_value == 'geofrgb':
        n_embed = 7
        model.placeholder = torch.nn.Parameter(torch.tensor(0.0))
    if args.ver_value == 'geof':
        n_embed = 4
        model.placeholder = torch.nn.Parameter(torch.tensor(0.0))

    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)
    if args.cuda:
        model.cuda()
    return model


def read_kitti360_poses(poses_path: str, T_velo2cam: np.ndarray) -> np.ndarray:
    """Read poses from poses.txt file. The poses are transformations from the velodyne coordinate
    system to the world coordinate system.
    :return: array of poses (Nx4x4), where N is the number of velodyne scans
    """

    # Load poses. Some poses are missing, because the camera was not moving.
    compressed_poses = np.loadtxt(poses_path, dtype=np.float32)
    frames = compressed_poses[:, 0].astype(np.int32)
    lidar_poses = compressed_poses[:, 1:].reshape(-1, 4, 4)

    # Create a full list of poses (with missing poses with value of the last known pose)
    sequence_length = np.max(frames) + 1
    poses = np.zeros((sequence_length, 4, 4), dtype=np.float32)

    last_valid_pose = lidar_poses[0]
    for i in range(sequence_length):
        if i in frames:
            last_valid_pose = lidar_poses[frames == i] @ T_velo2cam
        poses[i] = last_valid_pose
    return poses


def read_txt(file: str, sequence_name: str):
    with open(file, 'r') as f:
        lines = f.readlines()
    return [os.path.basename(line.strip()) for line in lines if sequence_name in line]


def get_window_range(path: str):
    file_name = os.path.basename(path)
    file_name = os.path.splitext(file_name)[0]
    interval = file_name.split('_')
    return int(interval[0]), int(interval[1])


def read_kitti360_scan(velodyne_path: str, i: int):
    file = os.path.join(velodyne_path, f'{i:010d}.bin')
    scan = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    return scan


def get_disjoint_ranges(paths: list[str]) -> list[tuple[int, int]]:
    """ Create list of disjoint ranges. """
    paths.sort()
    ranges = [get_window_range(path) for path in paths]
    disjoint_ranges = []
    for i, (start, end) in enumerate(ranges):
        if i == len(ranges) - 1:
            disjoint_ranges.append((start, end))
        else:
            next_start, _ = ranges[i + 1]
            if next_start < end:
                end = next_start - 1
            disjoint_ranges.append((start, end))
    return disjoint_ranges


def names_from_ranges(window_ranges: list[tuple[int, int]]) -> list[str]:
    """ Create list of window names. """
    return [f'{start:06d}_{end:06d}' for start, end in window_ranges]

# def get_list(ranges: list[tuple[int, int]]) -> list[int]:
#     """ Create list of sorted values without duplicates. """
#     values = []
#     for start, end in ranges:
#         values.extend(range(start, end))
#     return sorted(set(values))


# def split_indices(ranges: list[tuple[int, int]], all_indices: list) -> np.ndarray:
#     """ Create list of sorted indices without duplicates. """
#     values = []
#     for start, end in ranges:
#         values.extend(range(start, end))
#     indices = sorted(set(values))
#     return np.searchsorted(all_indices, indices)
