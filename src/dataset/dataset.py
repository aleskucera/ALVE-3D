import os
import logging

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .utils import open_sequence
from src.laserscan import LaserScan

log = logging.getLogger(__name__)


class SemanticDataset(Dataset):
    """ Wrapper class for the PyTorch Dataset class.

    :param dataset_path: Path to the dataset.
    :param cfg: Configuration object.
    :param sequences: List of sequences to load. If None all the sequences in the split are loaded (default: None)
    :param split: Split to load. If None all the sequences are loaded (default: None)
    :param size: Number of samples to load. If None all the samples are loaded (default: None)
    :param indices: List of indices to load. If None all the samples are loaded (default: None)
    """

    def __init__(self, dataset_path: str, cfg: DictConfig, sequences: list = None,
                 split: str = None, size: int = None, indices: list = None):

        self.cfg = cfg
        self.size = size
        self.split = split
        self.indices = indices
        self.path = dataset_path

        if sequences is None:
            sequences = cfg.split[split]
        self.sequences = sequences

        self.scans = []
        self.labels = []

        self.poses = []
        self.sequence_indices = []

        self.mean = np.array(cfg.mean, dtype=np.float32)
        self.std = np.array(cfg.std, dtype=np.float32)

        self.laser_scan = LaserScan(label_map=cfg.learning_map, colorize=False, color_map=cfg.color_map_train)

        self._init()

    def __getitem__(self, index):
        scan_path = self.scans[index]
        label_path = self.labels[index]

        # Load the scan data into the LaserScan object
        if self.split == 'train':
            self.laser_scan.open_scan(scan_path, flip_prob=0.5, trans_prob=0.5, rot_prob=0.5, drop_prob=0.5)
        else:
            self.laser_scan.open_scan(scan_path)

        # Load the label data into the LaserScan object
        self.laser_scan.open_label(label_path)

        if self.laser_scan.color is not None:
            # Concatenate depth, xyz, remission and color
            x = np.concatenate([self.laser_scan.proj_depth[np.newaxis, ...],
                                self.laser_scan.proj_xyz.transpose(2, 0, 1),
                                self.laser_scan.proj_remission[np.newaxis, ...],
                                self.laser_scan.proj_color.transpose(2, 0, 1)], axis=0)
        else:
            # Concatenate depth, xyz and remission
            x = np.concatenate([self.laser_scan.proj_depth[np.newaxis, ...],
                                self.laser_scan.proj_xyz.transpose(2, 0, 1),
                                self.laser_scan.proj_remission[np.newaxis, ...]], axis=0)

        # Normalize
        # x = (x - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]

        y = self.laser_scan.proj_sem_label.astype(np.long)

        return x, y, index

    def __len__(self):
        return len(self.scans)

    def _init(self):
        log.info(f"INFO: Initializing dataset (split: {self.split}) from path {self.path}")

        # ----------- LOAD -----------

        path = os.path.join(self.path, 'sequences')
        for seq in self.sequences:
            seq_path = os.path.join(path, f"{seq:02d}")
            points, labels, poses = open_sequence(seq_path, self.split)

            self.scans += points
            self.labels += labels
            self.poses += poses
            self.sequence_indices += [seq] * len(points)

        log.info(f"INFO: Found {len(self.scans)} samples")
        assert len(self.scans) == len(self.labels), "Number of points and labels must be equal"

        # ----------- CROP -----------

        self.scans = self.scans[:self.size]
        self.labels = self.labels[:self.size]

        log.info(f"INFO: Cropped dataset to {len(self.scans)} samples")

        # ----------- USE INDICES -----------

        if self.indices is not None:
            self._choose_data()
            log.info(f"INFO: Using samples {self.indices} for {self.split} split")

        log.info(f"INFO: Dataset initialized with {len(self.scans)} samples")

    def _choose_data(self):
        assert self.indices is not None

        self.indices.sort()

        assert max(self.indices) < len(self.scans), "Index out of range"

        self.scans = [self.scans[i] for i in self.indices]
        self.labels = [self.labels[i] for i in self.indices]
