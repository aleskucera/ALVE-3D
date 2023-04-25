import os

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from .utils import open_sequence
from src.utils.map import map_labels
from src.utils.cloud import transform_points, downsample_cloud, nearest_neighbors_2, \
    nearest_neighbors, nn_graph, connected_label_components

K_NN_ADJ = 5
K_NN_LOCAL = 20
VOXEL_SIZE = 0.2
STATIC_THRESHOLD = 0.2
DYNAMIC_THRESHOLD = 0.2


class SemanticKITTIConverter:
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.sequence = cfg.sequence if 'sequence' in cfg else 3

        old_sequence_path = os.path.join(cfg.ds.path, 'sequences_kitti', f"{self.sequence:02d}")
        self.sequence_dir = os.path.join(cfg.ds.path, 'sequences', f"{self.sequence:02d}")
        self.scans, self.labels, self.poses = open_sequence(old_sequence_path)

        self.window_ranges = create_window_ranges(self.scans)
        splits = get_splits(self.window_ranges, val_split=0.2)
        self.train_scans, self.val_scans, self.train_clouds, self.val_clouds = splits

    def convert(self):
        scans_dir = os.path.join(self.sequence_dir, 'velodyne')
        os.makedirs(scans_dir, exist_ok=True)

        clouds_dir = os.path.join(self.sequence_dir, 'voxel_clouds')
        os.makedirs(clouds_dir, exist_ok=True)

        clouds = np.sort(np.concatenate([self.train_clouds, self.val_clouds]))
        clouds = [os.path.join(clouds_dir, cloud) for cloud in clouds]

        val_scans, train_scans = self.val_scans.astype('S'), self.train_scans.astype('S')
        val_clouds, train_clouds = self.val_clouds.astype('S'), self.train_clouds.astype('S')

        with h5py.File(os.path.join(self.sequence_dir, 'info.h5'), 'w') as f:
            f.create_dataset('val', data=val_scans)
            f.create_dataset('train', data=train_scans)
            f.create_dataset('val_clouds', data=val_clouds)
            f.create_dataset('train_clouds', data=train_clouds)

        for cloud_file, window_range in zip(clouds, self.window_ranges):
            start, end = window_range
            global_points, global_labels = [], []
            for j in tqdm(range(start, end + 1), desc=f'Creating scans {start} - {end}'):
                scan_file = self.scans[j]
                label_file = self.labels[j]

                if scan_file.endswith('.bin') and label_file.endswith('.label'):
                    scan = np.fromfile(scan_file, dtype=np.float32)
                    scan = scan.reshape((-1, 4))
                    points = scan[:, :3]
                    remissions = scan[:, 3]

                    label = np.fromfile(label_file, dtype=np.int32)
                    semantics = label & 0xFFFF  # semantic label in lower half
                    semantics = semantics.flatten()

                    # Filter out the dynamic objects
                    dynamic_labels = [252, 253, 254, 255, 256, 257, 258, 259]
                    static_mask = np.isin(semantics, dynamic_labels, invert=True)
                    points = points[static_mask]
                    semantics = semantics[static_mask]
                    remissions = remissions[static_mask]

                    # Filter out the points which will be mapped to the ignore label
                    learn_labels = map_labels(semantics, self.cfg.ds.learning_map)
                    mask = learn_labels != self.cfg.ds.ignore_index
                    points = points[mask]
                    semantics = semantics[mask]
                    remissions = remissions[mask]

                    global_points.append(transform_points(points, self.poses[j]))
                    global_labels.append(semantics)
                    # Save the voxel map

                    # Write the scan to a file
                    with h5py.File(os.path.join(scans_dir, f'{j:06d}.h5'), 'w') as f:
                        f.create_dataset('points', data=points, dtype=np.float32)
                        f.create_dataset('labels', data=semantics, dtype=np.uint8)
                        f.create_dataset('pose', data=self.poses[j], dtype=np.float32)
                        f.create_dataset('remissions', data=remissions, dtype=np.float32)
                        # Save the voxel map

            # Create global point cloud
            global_points = np.concatenate(global_points)
            global_labels = np.concatenate(global_labels)

            # Downsample the point cloud
            voxel_points, voxel_labels = downsample_cloud(points=global_points, labels=global_labels, voxel_size=0.2)
            voxel_mask = np.zeros(len(voxel_points), dtype=np.bool)

            for j in tqdm(range(start, end + 1), desc=f'Determining voxel map {start} - {end}'):
                with h5py.File(os.path.join(scans_dir, f'{j:06d}.h5'), 'r+') as f:
                    points = np.asarray(f['points'])
                    transformed_points = transform_points(points, self.poses[j])
                    dists, voxel_indices = nearest_neighbors_2(voxel_points, transformed_points, k_nn=1)
                    f.create_dataset('voxel_map', data=voxel_indices.flatten(), dtype=np.int32)
                    voxel_mask[voxel_indices] = True

            filter_map = np.cumsum(voxel_mask) - 1
            for j in tqdm(range(start, end + 1), desc=f'Changing voxel maps {start} - {end}'):
                with h5py.File(os.path.join(scans_dir, f'{j:06d}.h5'), 'r+') as f:
                    voxel_map = np.asarray(f['voxel_map'])
                    f['voxel_map'][:] = filter_map[voxel_map]

            voxel_points = voxel_points[voxel_mask]
            voxel_labels = voxel_labels[voxel_mask]

            local_neighbors, _ = nearest_neighbors(voxel_points, K_NN_LOCAL)
            edge_sources, edge_targets, distances = nn_graph(voxel_points, K_NN_ADJ)
            objects = connected_label_components(voxel_labels, edge_sources, edge_targets)

            with h5py.File(cloud_file, 'w') as f:
                f.create_dataset('points', data=voxel_points, dtype='float32')

                f.create_dataset('objects', data=objects.flatten(), dtype='uint32')
                f.create_dataset('labels', data=voxel_labels.flatten(), dtype='uint8')

                f.create_dataset('local_neighbors', data=local_neighbors, dtype='uint32')
                f.create_dataset('edge_sources', data=edge_sources.flatten(), dtype='uint32')
                f.create_dataset('edge_targets', data=edge_targets.flatten(), dtype='uint32')


def create_window_ranges(scans: list, window_size: int = 200):
    """ Create a list of tuples containing the start and end indices of the windows.
    Example with window_size = 200 and 1100 scans:
    [(0, 199), (200, 399), (400, 599), (600, 799), (800, 999), (1000, 1099)]
    """
    window_ranges = []
    for i in range(0, len(scans), window_size):
        window_ranges.append((i, min(i + window_size - 1, len(scans) - 1)))
    return window_ranges


def get_splits(window_ranges: list, val_split: float = 0.2):
    """ Split the data into training and validation sets.
    The validation set are randomly selected windows, which sum is nearest to the val_split.

    Example:
    window_ranges = [(0, 199), (200, 399), (400, 599), (600, 799), (800, 999),
    (1000, 1199), (1200, 1399), (1400, 1599), (1600, 1799), (1800, 1999), (2000, 2023)]
    val_split = 0.2 -> 20% of the data is used for validation (0.2 * 2023 = 404.6 scans)
    selected_windows = [(0, 199), (800, 999)]

    val_scans = [000000.h5, 000001.h5, ..., 000199.h5, 000800.h5, 000801.h5, ..., 000999.h5]
    train_scans = [000200.h5, 000201.h5, ..., 000799.h5, 001000.h5, 001001.h5, ..., 002023.h5]
    val_clouds = [000000_000199.h5, 000800_000999.h5]
    train_clouds = [000200_000799.h5, 001000_002023.h5]
    """

    val_scans = np.array([], dtype=np.str_)
    train_scans = np.array([], dtype=np.str_)

    val_clouds = np.array([], dtype=np.str_)
    train_clouds = np.array([], dtype=np.str_)

    num_scans = sum([end - start + 1 for start, end in window_ranges])
    num_val_scans = int(num_scans * val_split)

    val_ranges = []
    train_ranges = window_ranges.copy()
    while num_val_scans > 0:
        idx = np.random.choice(len(train_ranges))
        rng = train_ranges[idx]
        val_ranges.append(rng)
        num_val_scans -= rng[1] - rng[0] + 1
        train_ranges.remove(rng)

    for rng in val_ranges:
        start, end = rng
        scan_names = [f'{i:06d}.h5' for i in range(start, end + 1)]
        val_scans = np.concatenate([val_scans, np.array(scan_names, dtype=np.str_)])
        val_clouds = np.append(val_clouds, f'{start:06d}_{end:06d}')

    for rng in train_ranges:
        start, end = rng
        scan_names = [f'{i:06d}.h5' for i in range(start, end + 1)]
        train_scans = np.concatenate([train_scans, np.array(scan_names, dtype=np.str_)])
        train_clouds = np.append(train_clouds, f'{start:06d}_{end:06d}')

    return train_scans, val_scans, train_clouds, val_clouds
