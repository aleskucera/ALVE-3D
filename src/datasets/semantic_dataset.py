import time

import h5py
import torch
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.project import project_scan
from .utils import update_selection_mask, load_semantic_dataset


class SemanticDataset(Dataset):
    """Semantic dataset for active learning.

    :param dataset_path: Path to the dataset.
    :param cfg: Configuration object.
    :param split: Split to load. ('train', 'val')
    :param size: Number of samples to load. If None all the samples are loaded (default: None)
    """

    def __init__(self, dataset_path: str, cfg: DictConfig, split: str, mode: str = 'passive', sequences: iter = None,
                 size: int = None):

        assert split in ['train', 'val']
        assert mode in ['passive', 'active']

        self.cfg = cfg
        self.mode = mode
        self.size = size
        self.split = split
        self.path = dataset_path

        if sequences is None:
            self.sequences = cfg.split[split]
        else:
            self.sequences = sequences

        self.proj_W = cfg.projection.W
        self.proj_H = cfg.projection.H
        self.proj_fov_up = cfg.projection.fov_up
        self.proj_fov_down = cfg.projection.fov_down

        self.poses = None
        self.scans = None
        self.labels = None

        self.cloud_map = None
        self.sample_map = None
        self.sequence_map = None
        self.selection_mask = None

        self._initialize()

    @property
    def scan_files(self):
        selected = np.where(self.selection_mask == 1)[0]
        return self.scans[selected]

    @property
    def label_files(self):
        selected = np.where(self.selection_mask == 1)[0]
        return self.labels[selected]

    def _initialize(self):
        data = load_semantic_dataset(self.path, self.sequences, self.split, self.mode)
        self.scans, self.labels, self.poses, self.sequence_map, self.cloud_map, self.selection_mask = data

        self.poses = self.poses[:self.size]
        self.scans = self.scans[:self.size]
        self.labels = self.labels[:self.size]
        self.sequence_map = self.sequence_map[:self.size]
        self.cloud_map = self.cloud_map[:self.size]
        self.selection_mask = self.selection_mask[:self.size]

        self.sample_map = np.arange(len(self.scans))

    def update(self):
        """Update the selection masks. This is used when the dataset is used in an active
        training loop after the new labels are available.
        """

        self.selection_mask = update_selection_mask(self.path, self.sequences, self.split)[:self.size]

    def label_voxels(self, voxels: np.ndarray, sequence: int, cloud: int):
        seq_indices = np.where(self.sequence_map == sequence)[0]

        seq_labels = self.labels[seq_indices]
        seq_cloud_map = self.cloud_map[seq_indices]
        seq_sample_map = self.sample_map[seq_indices]

        cloud_indices = np.where(seq_cloud_map == cloud)[0]

        cloud_labels = seq_labels[cloud_indices]
        cloud_sample_map = seq_sample_map[cloud_indices]

        for label_file, sample_idx in zip(cloud_labels, cloud_sample_map):
            with h5py.File(label_file, 'r+') as f:
                voxel_map = f['voxel_map']
                label_mask = f['label_mask']
                label_mask[np.isin(voxel_map, voxels)] = 1

                # Update the selection mask
                if np.sum(label_mask) > 0:
                    self.selection_mask[sample_idx] = 1

    def label_samples(self, sample_indices: np.ndarray):
        """Label the samples in the dataset. This is used when the dataset is used in an active
        training loop after the new labels are available.
        """

        self.selection_mask[sample_indices] = 1

        for idx in sample_indices:
            with h5py.File(self.labels[idx], 'r+') as f:
                label_mask = f['label_mask']
                label_mask[:] = 1

    def get_item(self, idx):
        # Load scan
        with h5py.File(self.scans[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions']).flatten()

        # Load label
        with h5py.File(self.labels[idx], 'r') as f:
            labels = np.asarray(f['labels']).flatten()
            voxel_map = np.asarray(f['voxel_map']).flatten()

        # Project points to image
        proj = project_scan(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_depth, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        proj_labels = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        proj_labels[proj_mask] = labels[proj_idx[proj_mask]]

        proj_image = np.concatenate([proj_depth[..., np.newaxis],
                                     proj_remissions[..., np.newaxis],
                                     proj_colors], axis=-1, dtype=np.float32)

        proj_voxel_map = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        proj_voxel_map[proj_mask] = voxel_map[proj_idx[proj_mask]]

        proj_image = torch.from_numpy(proj_image).permute(2, 0, 1)
        proj_labels = torch.from_numpy(proj_labels)
        proj_voxel_map = torch.from_numpy(proj_voxel_map)

        return proj_image, proj_labels, proj_voxel_map, self.sequence_map[idx], self.cloud_map[idx]

    def __getitem__(self, idx):
        # Load scan
        with h5py.File(self.scan_files[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions']).flatten()

        # Load label
        with h5py.File(self.label_files[idx], 'r') as f:
            labels = np.asarray(f['labels']).flatten()
            label_mask = np.asarray(f['label_mask']).flatten()

        # Apply mask
        labels *= label_mask

        # Project points to image
        proj = project_scan(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_depth, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        proj_labels = np.zeros((self.proj_H, self.proj_W), dtype=np.long)
        proj_labels[proj_mask] = labels[proj_idx[proj_mask]]

        proj_image = np.concatenate([proj_depth[..., np.newaxis],
                                     proj_remissions[..., np.newaxis],
                                     proj_colors], axis=-1, dtype=np.float32)

        proj_image = proj_image.transpose((2, 0, 1))

        return proj_image, proj_labels

    def __len__(self):
        return np.sum(self.selection_mask)

    def get_true_length(self):
        return len(self.scans)

    def __str__(self):
        return f'\nSemanticDataset: {self.split}\n' \
               f'\t - Dataset size: {len(self)}\n' \
               f'\t - Sequences: {self.sequences}\n'
