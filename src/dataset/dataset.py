import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset
from copy import deepcopy
import open3d as o3d
from tqdm import tqdm

from .ds_utils import Sample, \
    create_semantic_laser_scan, \
    init_sequences, read_sequence_data, \
    transform_points, apply_augmentations, map_colors, map_labels


class SemanticDataset(Dataset):
    def __init__(self, path: str, cfg: DictConfig, split: str = None):
        """ Initialize the dataset
        :param path: path to the dataset (the directory containing the sequences)
        :param split: train, val or test
        :param cfg: configuration
        """

        self.cfg = cfg
        self.path = path
        self.split = split

        # Create scan object for reading the data
        self.scan = create_semantic_laser_scan(cfg)

        # Initialize the list of sequences to load
        self.sequences = init_sequences(self.path, cfg.split[split], cfg.sequence_structure)

        # Get the list of samples from the sequences
        self.samples = self.get_samples(self.sequences)

    @staticmethod
    def get_samples(sequences: list) -> list:
        samples = []
        for seq in sequences:
            ids, points, labels, calib, poses, times = read_sequence_data(seq)
            for i in ids:
                sample = Sample(id=i, time=times[i], points_path=points[i], label_path=labels[i],
                                calibration=calib, pose=poses[i])
                samples.append(sample)
        return samples

    def __getitem__(self, index):

        # Load the sample
        sample = deepcopy(self.samples[index])
        data, label = self.load_sample_data(sample)
        label = map_labels(label, self.cfg.learning_map)

        # Apply augmentations
        if self.split == 'train':
            data, label = apply_augmentations(data, label)

        sample.points, sample.label = data, label
        return sample

    def load_sample_data(self, sample: Sample):
        # Load the sample
        self.scan.open_scan(sample.points_path)
        self.scan.open_label(sample.label_path)

        # Create data
        xyz = self.scan.proj_xyz.transpose([2, 0, 1])  # (3 x H x W)
        intensity = self.scan.proj_remission[np.newaxis, ...]  # (1 x H x W)
        depth = self.scan.proj_range[np.newaxis, ...]  # (1 x H x W)
        data = np.concatenate([xyz, intensity, depth], axis=0)  # (5 x H x W)

        # Create label
        label = self.scan.proj_sem_label.copy()

        return data, label

    def create_global_cloud(self, sequence: int, step: int = 50) -> o3d.geometry.PointCloud:
        """ Create a global point cloud from the sequence
        :param sequence: sequence index
        :param step: step between two points
        :return: point cloud
        """
        colors = []
        global_cloud = []

        # Load samples from the sequence
        seq = init_sequences(self.path, [sequence], self.cfg.sequence_structure)[0]
        samples = self.get_samples([seq])

        # Loop through the samples and store the points and their colors
        for sample in tqdm(samples[::step]):
            data, label = self.load_sample_data(sample)

            # Transform the points
            cloud = data[:3, ...].transpose([1, 2, 0]).reshape([-1, 3])
            cloud = transform_points(cloud, sample.pose)
            global_cloud.append(cloud)

            # Map the colors
            color = map_colors(label, self.cfg.color_map).reshape([-1, 3])
            colors.append(color)

        # Concatenate the points and colors
        global_cloud = np.concatenate(global_cloud, axis=0)
        colors = np.concatenate(colors, axis=0)

        # Create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(global_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def __len__(self):
        return len(self.samples)
