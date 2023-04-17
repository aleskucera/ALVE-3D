import os

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from .utils import open_sequence


class SemanticKITTIConverter:
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.sequence = cfg.sequence if 'sequence' in cfg else 3

        self.sequence_path = os.path.join(cfg.ds.path, 'sequences', f"{self.sequence:02d}")
        self.scans, self.labels, self.poses = open_sequence(self.sequence_path)

        self.window_ranges = create_window_ranges(self.scans)
        splits = self.get_splits(self.window_ranges)
        self.train_scans, self.val_scans, self.train_clouds, self.val_clouds = splits

    def convert(self):
        scans_dir = os.path.join(self.sequence_path, 'velodyne')
        os.makedirs(scans_dir, exist_ok=True)

        clouds_dir = os.path.join(self.sequence_path, 'voxel_clouds')
        os.makedirs(clouds_dir, exist_ok=True)

        clouds = np.sort(np.concatenate([self.train_clouds, self.val_clouds]))
        clouds = [os.path.join(clouds_dir, cloud) for cloud in clouds]

        val_scans, train_scans = val_scans.astype('S'), train_scans.astype('S')
        val_clouds, train_clouds = val_clouds.astype('S'), train_clouds.astype('S')

        with h5py.File(os.path.join(self.sequence_path, 'info.h5'), 'w') as f:
            f.create_dataset('val', data=val_samples)
            f.create_dataset('train', data=train_samples)
            f.create_dataset('val_clouds', data=val_clouds)
            f.create_dataset('train_clouds', data=train_clouds)

        for cloud, window_range in zip(clouds, self.window_ranges):

            start, end = window_range
            for j in tqdm(range(start, end + 1), desc=f'Creating scans {start} - {end}'):
                # Load scans and create global cloud
                # Also save the scans to disk

                # Write the scan to a file
                with h5py.File(os.path.join(scans_dir, f'{j:06d}.h5'), 'w') as f:
                    f.create_dataset('pose', data=self.poses[j], dtype=np.float32)
                    f.create_dataset('colors', data=colors, dtype=np.float32)
                    f.create_dataset('points', data=scan_points, dtype=np.float32)
                    f.create_dataset('remissions', data=scan_remissions, dtype=np.float32)

                    f.create_dataset('labels', data=labels, dtype=np.uint8)
                    f.create_dataset('voxel_map', data=voxel_indices.astype(np.uint32), dtype=np.uint32)

            # Create global cloud
            # Update voxel masks


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

    val_windows = []
    while num_val_scans > 0:
        window = np.random.choice(window_ranges)
        val_windows.append(window)
        num_val_scans -= window[1] - window[0] + 1
        window_ranges.remove(window)

    for window in val_windows:
        start, end = window
        scan_names = [f'{i:06d}.h5' for i in range(start, end + 1)]
        val_scans = np.concatenate([val_scans, np.array(scan_names, dtype=np.str_)])
        val_clouds = np.append(val_clouds, f'{start:06d}_{end:06d}')

    for window in window_ranges:
        start, end = window
        scan_names = [f'{i:06d}.h5' for i in range(start, end + 1)]
        train_scans = np.concatenate([train_scans, np.array(scan_names, dtype=np.str_)])
        train_clouds = np.append(train_clouds, f'{start:06d}_{end:06d}')

    return train_scans, val_scans, train_clouds, val_clouds
