import os
import logging

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset


class PartitionDataset(Dataset):
    def __init__(self, dataset_path: str, project_name: str, cfg: DictConfig, split: str, size: int = None,
                 al_experiment: bool = False, sequences: iter = None, resume: bool = False):

        assert split in ['train', 'val']

        self.cfg = cfg
        self.size = size
        self.resume = resume
        self.split = split
        self.path = dataset_path
        self.project_name = project_name
        self.al_experiment = al_experiment

        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = cfg.split[split]

        self.cloud_files = None
        self.selection_mask = None

        self._initialize()

    @property
    def clouds(self):
        selected = np.where(self.selection_mask == 1)[0]
        return self.cloud_files[selected]

    def _initialize(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        return len(self.clouds)
