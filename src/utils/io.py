import os

import h5py
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def set_paths(cfg: DictConfig, output_dir: str) -> DictConfig:
    cfg.path.output = output_dir
    for i in cfg.path:
        cfg.path[i] = to_absolute_path(cfg.path[i])
    cfg.ds.path = to_absolute_path(cfg.ds.path)
    return cfg


def get_split(dataset_path: str, sequences: list, split: str) -> tuple:
    assert split in ['train', 'val']
    for sequence in sequences:
        velodyne_path = os.path.join(os.path.join(dataset_path, 'velodyne'))
        labels_path = os.path.join(os.path.join(dataset_path, 'labels'))
        info_path = os.path.join(dataset_path, 'sequences', f'{sequence:02d}', 'info.txt')

        velodyne = [os.path.join(velodyne_path, v) for v in os.listdir(velodyne_path)]
        velodyne.sort()

        labels = [os.path.join(labels_path, l) for l in os.listdir(labels_path)]
        labels.sort()

        with h5py.File(info_path, 'r') as f:
            poses = np.asarray(f['poses'])
            indices = np.asarray(f[split])
            selected = np.asarray(f['selected']) if split == 'train' else None

        velodyne = [velodyne[i] for i in indices]
        labels = [labels[i] for i in indices]

        yield velodyne, labels, poses, selected
