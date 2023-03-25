import os

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils import project_points, map_labels, augment_points
from .utils import load_semantic_dataset


class SelectionDataset(Dataset):
    """PyTorch dataset wrapper for the semantic segmentation dataset. This dataset is used also for active learning.


    :param dataset_path: Path to the dataset.
    :param cfg: Configuration of the dataset.
    :param split: Split to load. ('train', 'val')
    :param size: Number of samples to load. If None all the samples are loaded (default: None)
    :param active_mode: If True the dataset initializes in active
                                 mode - deactivates all labels. (default: False)
    :param sequences: List of sequences to load. If None all the sequences defined in config
                      for a given split are loaded. (default: None)
    """

    def __init__(self, dataset_path: str, cfg: DictConfig, split: str, size: int = None,
                 active_mode: bool = False, sequences: iter = None):

        assert split in ['train', 'val']

        self.cfg = cfg
        self.size = size
        self.split = split
        self.path = dataset_path
        self.active = active_mode
        self.label_map = cfg.learning_map
        self.num_classes = cfg.num_classes
        self.ignore_index = cfg.ignore_index

        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = cfg.split[split]

        self.proj_W = cfg.projection.W
        self.proj_H = cfg.projection.H
        self.proj_fov_up = cfg.projection.fov_up
        self.proj_fov_down = cfg.projection.fov_down

        self.scans = None
        self.labels = None

        self.cloud_map = None
        self.sample_map = None
        self.sequence_map = None
        self.selection_mask = None

        self._initialize()

    def _initialize(self) -> None:
        """ Initialize the dataset. Sets the sample selection masks and sample masks to 0 or 1 based on the
        active learning mode. Then loads the following data:

        - scans: Array of paths to scans. (N,)
        - labels: Array of paths to labels. (N,)
        - poses: Array of poses. (N, 4, 4)
        - sequence_map: Array of sequence indices that maps each sample to a sequence. (N,)
        - cloud_map: Array of cloud indices that maps each sample to a global cloud. (N,)
        - selection_mask: Array of 0 or 1 that indicates if the sample is selected or not. (N,)

        Then the function crops the dataset to the desired size (self.size) and creates a sample map that maps
        the samples to the original dataset.
        """

        data = load_semantic_dataset(self.path, self.sequences, self.split, self.active, selection=True)
        self.scans, self.labels, self.sequence_map, self.cloud_map, self.selection_mask = data

        self.scans = self.scans[:self.size]
        self.labels = self.labels[:self.size]
        self.sequence_map = self.sequence_map[:self.size]
        self.cloud_map = self.cloud_map[:self.size]
        self.selection_mask = self.selection_mask[:self.size]

        self.sample_map = np.arange(len(self.scans))

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """ Return a sample from the dataset. Typically used in an active learning loop for
        selecting samples to label. The difference between this function and __getitem__ is that
        this function returns the following data:

        - Projected scan to the 2D plane (5, H, W)
        - Projected radial distances to the 2D plane (H, W)
        - Projected voxel map to the 2D plane (H, W)
        - Sequence index
        - Cloud index

        This function does not apply augmentations and does not return labels.

        :param idx: Sample index.
        :return: Projected scan, projected voxel map, sequence index, cloud index.
        """

        # Load scan
        with h5py.File(self.scans[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions']).flatten()

        # Load voxel map
        with h5py.File(self.labels[idx], 'r') as f:
            labels = np.asarray(f['labels']).flatten()
            voxel_map = np.asarray(f['voxel_map']).flatten().astype(np.float32)
            voxel_map[labels == self.ignore_index] = -1

        # Project points to image and map the projection
        proj = project_points(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_distances, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        # Project remissions
        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        # Project colors
        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        # Concatenate scan features
        proj_scan = np.concatenate([proj_distances[..., np.newaxis],
                                    proj_remissions[..., np.newaxis],
                                    proj_colors], axis=-1, dtype=np.float32).transpose((2, 0, 1))

        # Project voxel map
        proj_voxel_map = np.full((self.proj_H, self.proj_W), -1, dtype=np.int64)
        proj_voxel_map[proj_mask] = voxel_map[proj_idx[proj_mask]]

        return proj_scan, proj_distances, proj_voxel_map, self.cloud_map[idx]

    def __len__(self):
        return len(self.scans)

    def get_dataset_clouds(self) -> np.ndarray:
        return np.unique(self.cloud_map)

    def __str__(self):
        return f'\nSelectionDataset: {self.split}\n' \
               f'\t - Dataset size: {len(self)}\n' \
               f'\t - Sequences: {self.sequences}\n'
