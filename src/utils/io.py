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


def load_dataset(dataset_path: str, sequences: list, split: str) -> tuple:
    assert split in ['train', 'val']

    poses = []
    labels = []
    velodyne = []
    cloud_maps = []
    selection_masks = []

    for sequence in sequences:
        sequence_path = os.path.join(dataset_path, 'sequences', f'{sequence:02d}')

        info_path = os.path.join(sequence_path, 'info.h5')
        labels_path = os.path.join(os.path.join(sequence_path, 'labels'))
        velodyne_path = os.path.join(os.path.join(sequence_path, 'velodyne'))

        with h5py.File(info_path, 'r') as f:
            split_indices = np.asarray(f[split])

            print(
                f'Split indices max: {np.max(split_indices)}, min: {np.min(split_indices)}, shape: {split_indices.shape}')
            print(f'Poses shape: {np.asarray(f["poses"]).shape}')
            print(f'Cloud map shape: {np.asarray(f["cloud_map"]).shape}')

            poses.append(np.asarray(f['poses'])[split_indices])
            cloud_maps.append(np.asarray(f['cloud_map'])[split_indices])

            if split == 'train':
                selection_masks.append(np.asarray(f['selection_mask']))

        seq_labels = [os.path.join(labels_path, l) for l in os.listdir(labels_path) if l.endswith('.h5')]
        seq_velodyne = [os.path.join(velodyne_path, v) for v in os.listdir(velodyne_path) if v.endswith('.h5')]

        seq_labels.sort()
        seq_velodyne.sort()

        labels.append(np.array(seq_labels, dtype=np.str_)[split_indices])
        velodyne.append(np.array(seq_velodyne, dtype=np.str_)[split_indices])

    return velodyne, labels, poses, cloud_maps, selection_masks


def update_selection_mask(dataset_path: str, sequences: list, split: str) -> list:
    assert split in ['train', 'val']
    selection_masks = []
    for sequence in sequences:
        info_path = os.path.join(dataset_path, 'sequences', f'{sequence:02d}', 'info.h5')
        with h5py.File(info_path, 'r') as f:
            seq_selection_mask = np.asarray(f['selection_mask'])
        selection_masks.append(seq_selection_mask)
    return selection_masks
