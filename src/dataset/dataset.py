import os
import logging

import numpy as np

from torch.utils.data import Dataset
from omegaconf import DictConfig
from src.laserscan import SemLaserScan
from .utils import dict_to_label_map, open_sequence

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

        self.label_map = dict_to_label_map(cfg.learning_map)
        self.scan = SemLaserScan()

        self.init()

    def __getitem__(self, index):
        points_path = self.points[index]
        label_path = self.labels[index]

        self.scan.open_scan(points_path)
        self.scan.open_label(label_path)

        image = self.scan.proj_depth[np.newaxis, ...]
        label = self.label_map[self.scan.proj_sem_label].astype(np.long)

        # TODO: Add augmentation

        return image, label, index

    def __len__(self):
        return len(self.points)

    def init(self):
        log.info(f"Initializing dataset from path {self.path}")

        # ----------- LOAD -----------

        path = os.path.join(self.path, 'sequences')
        for seq in self.sequences:
            seq_path = os.path.join(path, f"{seq:02d}")
            points, labels = open_sequence(seq_path)
            self.points += points
            self.labels += labels

        log.info(f"Found {len(self.points)} samples")
        assert len(self.points) == len(self.labels), "Number of points and labels must be equal"

        # ----------- CROP -----------

        self.points = self.points[:self.size]
        self.labels = self.labels[:self.size]

        log.info(f"Cropped dataset to {len(self.points)} samples")

        # ----------- USE INDICES -----------

        if self.indices is not None:
            self.indices.sort()
            log.info(f"Using samples {self.indices} for {self.split} split")

            assert max(self.indices) < len(self.points), "Index out of range"

            self.points = [self.points[i] for i in self.indices]
            self.labels = [self.labels[i] for i in self.indices]

        log.info(f"Dataset initialized with {len(self.points)} samples")
