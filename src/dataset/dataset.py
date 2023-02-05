import os
import logging

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .utils import open_sequence
from src.laserscan import LaserScan

log = logging.getLogger(__name__)


class SemanticDataset(Dataset):
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

        self.points = []
        self.labels = []
        self.poses = []
        self.sequence_indices = []

        self.mean = np.array(cfg.mean, dtype=np.float32)
        self.std = np.array(cfg.std, dtype=np.float32)

        self.scan = LaserScan(label_map=cfg.learning_map, colorize=False, color_map=cfg.color_map_train)

        self.init()

    def __getitem__(self, index):
        points_path = self.points[index]
        label_path = self.labels[index]

        if self.split == 'train':
            self.scan.open_scan(points_path, flip_prob=0.5, trans_prob=0.5, rot_prob=0.5, drop_prob=0.5)
        else:
            self.scan.open_scan(points_path)

        self.scan.open_label(label_path)

        # Concatenate depth, xyz and remission
        proj = np.concatenate([self.scan.proj_depth[np.newaxis, ...],
                               self.scan.proj_xyz.transpose(2, 0, 1),
                               self.scan.proj_remission[np.newaxis, ...]], axis=0)

        # Normalize
        proj = (proj - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]

        label = self.scan.proj_sem_label.astype(np.long)

        return proj, label, index

    def __len__(self):
        return len(self.points)

    def init(self):
        log.info(f"Initializing dataset from path {self.path}")

        # ----------- LOAD -----------

        path = os.path.join(self.path, 'sequences')
        for seq in self.sequences:
            seq_path = os.path.join(path, f"{seq:02d}")
            points, labels, poses = open_sequence(seq_path, self.split)
            self.points += points
            self.labels += labels
            self.poses += poses
            self.sequence_indices += [seq] * len(points)

        log.info(f"Found {len(self.points)} samples")
        assert len(self.points) == len(self.labels), "Number of points and labels must be equal"

        # ----------- CROP -----------

        self.points = self.points[:self.size]
        self.labels = self.labels[:self.size]

        log.info(f"Cropped dataset to {len(self.points)} samples")

        # ----------- USE INDICES -----------

        if self.indices is not None:
            self.choose_data()
            log.info(f"Using samples {self.indices} for {self.split} split")

        log.info(f"Dataset initialized with {len(self.points)} samples")

    def choose_data(self, indices=None):
        if indices:
            self.indices = indices
        assert self.indices is not None

        self.indices.sort()

        assert max(self.indices) < len(self.points), "Index out of range"

        self.points = [self.points[i] for i in self.indices]
        self.labels = [self.labels[i] for i in self.indices]
