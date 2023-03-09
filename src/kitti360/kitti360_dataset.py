import os
import logging

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset
from numpy.lib.recfunctions import structured_to_unstructured

from .ply import read_ply

log = logging.getLogger(__name__)


class KITTI360Dataset(Dataset):
    def __init__(self, dataset_path: str, cfg: DictConfig, split: str, size: int = None):
        self.cfg = cfg
        self.size = size
        self.split = split
        self.path = dataset_path

        self._init()

    def __getitem__(self, idx):
        scan_path = self.scans[idx]
        scan = read_ply(scan_path)

        points = structured_to_unstructured(scan[['x', 'y', 'z']])
        colors = structured_to_unstructured(scan[['red', 'green', 'blue']]) / 255
        x = np.concatenate([points, colors], axis=1)

        y = structured_to_unstructured(scan[['semantic']])

        return x, y

    def __len__(self):
        return len(self.scans)

    def _init(self):
        split_list = os.path.join(self.path, 'data_3d_semantics', 'train', f'2013_05_28_drive_{self.split}.txt')

        assert os.path.exists(split_list), f'Could not find split list {split_list}'
        with open(split_list, 'r') as f:
            scans = f.read().splitlines()

        self.scans = [os.path.join(self.path, scan) for scan in scans][:self.size]

        log.info(f'Loaded {len(self.scans)} scans from {self.split} split')
