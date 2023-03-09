import os
import logging

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .utils import open_sequence
from src.laserscan import LaserScan

log = logging.getLogger(__name__)


class SemanticKITTIDataset(Dataset):
    """ Wrapper class for the PyTorch Dataset class.

    :param dataset_path: Path to the dataset.
    :param cfg: Configuration object.
    :param sequences: List of sequences to load. If None all the sequences in the split are loaded (default: None)
    :param split: Split to load. If None all the sequences are loaded (default: None)
    :param size: Number of samples to load. If None all the samples are loaded (default: None)
    :param indices: List of indices to load. If None all the samples are loaded (default: None)
    """

    def __init__(self, dataset_path: str, cfg: DictConfig, split: str,
                 sequences: list = None, size: int = None, indices: list = None):

        self.cfg = cfg
        self.size = size
        self.split = split
        self.indices = indices
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

    def calculate_statistics(self):
        """ Calculate the mean, standard deviation and ratio of the labels in the dataset. """
        log.info('Calculating statistics...')
        num_channels = self.cfg.num_semantic_channels
        num_classes = self.cfg.num_classes

        # Initialize variables
        num_pixels = 0
        mean = np.zeros(num_channels)
        std = np.zeros(num_channels)
        content = np.zeros(num_classes)

        log.info(f'Calculating mean and content ratio...')
        for i in tqdm(range(len(self))):
            proj_image, proj_label, idx = self[i]

            # Update the statistics
            num_pixels += proj_image.shape[1] * proj_image.shape[2]
            mean += np.sum(proj_image, axis=(1, 2))
            content += np.bincount(proj_label.flatten(), minlength=num_classes)

        # Calculate the mean
        mean /= num_pixels

        # Calculate the content ratio
        content[self.cfg.ignore_index] = 0
        ratio = content / np.sum(content)

        log.info(f'Calculating std...')
        for i in tqdm(range(len(self))):
            proj_image, proj_label, idx = self[i]

            # Update the statistics
            std += np.sum((proj_image - mean[:, np.newaxis, np.newaxis]) ** 2, axis=(1, 2))

        # Calculate the std
        std = np.sqrt(std / num_pixels)

        # Print the statistics
        log.info(f"Mean: {mean}")
        log.info(f"Std: {std}")
        log.info(f"Ratio: {ratio}")

        return mean, std, ratio

    def __getitem__(self, index):
        scan_path = self.scans[index]
        label_path = self.labels[index]

        if self.split == 'train':
            self.laser_scan.open_scan(scan_path, flip_prob=0.5, trans_prob=0.5, rot_prob=0.5, drop_prob=0.5)
        else:
            self.laser_scan.open_scan(scan_path)
        self.laser_scan.open_label(label_path)

        if self.laser_scan.color is not None:
            proj_image = np.concatenate([self.laser_scan.proj_color.transpose(2, 0, 1),
                                         self.laser_scan.proj_depth[np.newaxis, ...],
                                         self.laser_scan.proj_remission[np.newaxis, ...]
                                         ], axis=0)
        else:
            proj_image = np.concatenate([self.laser_scan.proj_depth[np.newaxis, ...],
                                         # self.laser_scan.proj_xyz.transpose(2, 0, 1),
                                         self.laser_scan.proj_remission[np.newaxis, ...]], axis=0)

        # Normalize
        # x = (x - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]

        proj_label = self.laser_scan.proj_sem_label.astype(np.long)

        return proj_image, proj_label, index

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
