import os
import logging

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from semantickitti.utils import open_sequence
from src.laserscan import LaserScan

log = logging.getLogger(__name__)


class SemanticKITTIDataset(Dataset):
    def __init__(self, dataset_path: str, cfg: DictConfig, split: str, sequences: list = None):

        self.cfg = cfg
        self.split = split
        self.path = dataset_path

        split_sequences = cfg.split[split]
        if sequences is None:
            self.sequences = split_sequences
        else:
            self.sequences = [s for s in sequences if s in split_sequences]

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

        if self.split == 'train':
            self.laser_scan.open_scan(scan_path, flip_prob=0.5, trans_prob=0.5, rot_prob=0.5, drop_prob=0.5)
        else:
            self.laser_scan.open_scan(scan_path)

        self.laser_scan.open_label(label_path)

        proj_scan = np.concatenate([self.laser_scan.proj_depth[np.newaxis, ...],
                                    # self.laser_scan.proj_xyz.transpose(2, 0, 1),
                                    self.laser_scan.proj_remission[np.newaxis, ...]], axis=0)

        proj_scan = (proj_scan - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]
        proj_label = self.laser_scan.proj_sem_label.astype(np.long)
        return proj_scan, proj_label

    def __len__(self):
        return len(self.scans)

    def _init(self):
        log.info(f"INFO: Initializing dataset (split: {self.split}) from path {self.path}")

        path = os.path.join(self.path, 'sequences')
        for seq in self.sequences:
            seq_path = os.path.join(path, f"{seq:02d}")
            points, labels, poses = open_sequence(seq_path)

            self.scans += points
            self.labels += labels
            self.poses += poses
            self.sequence_indices += [seq] * len(points)

        assert len(self.scans) == len(self.labels), "Number of points and labels must be equal"
        log.info(f"INFO: Dataset initialized with {len(self.scans)} samples")
