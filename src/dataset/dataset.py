import os

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .sequence import Sequence
from .laserscan import SemLaserScan


class SemanticDataset(Dataset):
    def __init__(self, path: str, cfg: DictConfig, split: str = None, size: int = None):
        """ Initialize the dataset
        :param path: path to the dataset (the directory containing the sequences)
        :param split: train, val or test
        :param cfg: configuration
        """

        self.cfg = cfg
        self.path = path
        self.size = size
        self.split = split

        # Create scan object for reading the data
        self.scan = _create_semantic_laser_scan(cfg)

        # Initialize the list of sequences to load
        self.sequences = _init_sequences(self.path, cfg.split[split], cfg.sequence_structure)

        # Get the dict of samples from the sequences
        self.samples = _get_samples(self.sequences, size=size)

    def __getitem__(self, index):
        sample = deepcopy(self.samples[index])
        sample.load_learning_data(self.scan, self.cfg.learning_map)

        # Apply augmentations
        # if self.split == 'train':
        #     sample.augment()
        return sample.x, sample.y

    def get_sem_cloud(self, index):
        """ Get the semantic point cloud for visualization
        :param index: index of the sample
        :return: the semantic point cloud sample
        """
        sample = deepcopy(self.samples[index])
        sample.load_semantic_cloud(self.scan)
        return sample

    def get_sem_depth(self, index):
        """ Get the semantic depth image for visualization
        :param index: index of the sample
        :return: the semantic depth image sample
        """
        sample = deepcopy(self.samples[index])
        sample.load_semantic_depth(self.scan)
        return sample

    def create_global_cloud(self, sequence_index: int, step: int = 50) -> tuple:
        """ Create a global point cloud from the sequence
        :param sequence_index: sequence index
        :param step: step between two points
        :return: point cloud
        """
        colors = []
        global_cloud = []

        # Load samples from the sequence
        seq = self.sequences[sequence_index]
        samples = _get_samples([seq], step=step)

        # Loop through the samples and store the points and their colors
        for s in tqdm(samples.values()):
            sample = deepcopy(s)
            sample.load_semantic_cloud(self.scan)

            # Transform the points
            sample.to_absolute_position()
            global_cloud.append(sample.points)
            colors.append(sample.colors)

        # Concatenate the points and colors
        global_cloud = np.concatenate(global_cloud, axis=0)
        colors = np.concatenate(colors, axis=0)
        return global_cloud, colors

    def __len__(self):
        return len(self.samples)


def _create_semantic_laser_scan(cfg: DictConfig) -> SemLaserScan:
    """ Create a semantic laser scan object
    :param cfg: configuration
    :return: semantic laser scan object
    """
    scan = SemLaserScan(
        nclasses=len(cfg.labels),
        sem_color_dict=cfg.color_map,
        project=True,
        H=cfg.laser_scan.H,
        W=cfg.laser_scan.W,
        fov_up=cfg.laser_scan.fov_up,
        fov_down=cfg.laser_scan.fov_down)
    return scan


def _init_sequences(path: str, seq_list: list, seq_structure: DictConfig) -> list:
    """ Initialize the sequences
    :param path: path to the sequences
    :param seq_list: list of sequences to load
    :param seq_structure: structure of the sequences
    :return: list of Sequence objects
    """
    sequences = []
    for seq in seq_list:
        seq_name = f"{seq:02d}"
        seq_path = os.path.join(path, seq_name)
        sequences.append(Sequence(name=seq_name,
                                  path=seq_path,
                                  points_dir=os.path.join(seq_path, seq_structure.points_dir),
                                  labels_dir=os.path.join(seq_path, seq_structure.labels_dir),
                                  calib_file=os.path.join(seq_path, seq_structure.calib_file),
                                  poses_file=os.path.join(seq_path, seq_structure.poses_file),
                                  times_file=os.path.join(seq_path, seq_structure.times_file)))
    return sequences


def _get_samples(sequences: list, size: int = None, step: int = 1, ) -> dict:
    """ Get the samples from the sequences and store them in a dict
    :param sequences: list of Sequence objects
    :param step: step between two samples (for subsampling)
    :return: dict of samples
    """
    samples, samples_list, index = {}, [], 0
    size = size if size is not None else float('inf')
    # Load the samples from the sequences
    for seq in sequences:
        samples_list += seq.get_samples()

    # Subsample the samples and store them in a dict until the size is reached
    for s in samples_list[::step]:
        samples[index] = s
        index += 1
        if index >= size:
            return samples

    return samples
