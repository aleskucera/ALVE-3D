import os
import logging

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils import project_points, map_labels, augment_points
from old.utils import load_semantic_dataset

log = logging.getLogger(__name__)


class SemanticDataset(Dataset):
    def __init__(self, dataset_path: str, project_name: str, cfg: DictConfig, split: str,
                 size: int = None, selection_mode: bool = False, al_experiment: bool = False,
                 sequences: iter = None, resume: bool = False):

        assert split in ['train', 'val']

        self.cfg = cfg
        self.size = size
        self.resume = resume
        self.split = split
        self.path = dataset_path
        self.project_name = project_name
        self.al_experiment = al_experiment
        self.selection_mode = selection_mode

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

        self.scan_files = None
        self.label_files = None

        self.cloud_map = None
        self.sample_map = None
        self.sequence_map = None
        self.selection_mask = None

        self._initialize()

    @property
    def scans(self) -> np.ndarray:
        if self.selection_mode:
            return self.scan_files
        selected = np.where(self.selection_mask == 1)[0]
        return self.scan_files[selected]

    @property
    def labels(self) -> np.ndarray:
        if self.selection_mode:
            return self.label_files
        selected = np.where(self.selection_mask == 1)[0]
        return self.label_files[selected]

    @property
    def voxel_clouds(self) -> np.ndarray:
        return np.unique(self.cloud_map)

    def get_cloud_id(self, idx: int) -> int:
        cloud = self.cloud_map[idx]
        return np.where(self.voxel_clouds == cloud)[0][0]

    def end_of_voxel_cloud(self, idx: int) -> bool:
        cloud = self.cloud_map[idx]
        return np.where(self.cloud_map == cloud)[0][-1] == idx

    def _initialize(self) -> None:
        data = load_semantic_dataset(self.path, self.project_name, self.sequences, self.split, self.al_experiment,
                                     self.resume)
        self.scan_files, self.label_files, self.sequence_map, self.cloud_map, self.selection_mask = data

        self.scan_files = self.scan_files[:self.size]
        self.label_files = self.label_files[:self.size]
        self.sequence_map = self.sequence_map[:self.size]
        self.cloud_map = self.cloud_map[:self.size]
        self.selection_mask = self.selection_mask[:self.size]

        self.sample_map = np.arange(len(self.scans))

        log.info(self.__repr__())

    def update_sequence_selection_masks(self) -> None:
        """ Update the selection masks for each sequence. This is used when the dataset is used in an active learning
        and the Selector object selects the samples to label. It updates the selection masks for each sequence based
        on the selection mask stored in the dataset. (self.selection_mask)
        """

        # Get unique sequences and their counts which are used to split the selection mask
        sequences, count = np.unique(self.sequence_map, return_counts=True)

        # Sort selection mask by sequence map and split it by sequence
        selection_mask = self.selection_mask[np.argsort(self.sequence_map)]
        selection_masks = np.split(selection_mask, np.cumsum(count)[:-1])

        # Update selection masks for each sequence
        for sequence, mask in zip(sequences, selection_masks):
            info_file = os.path.join(self.path, self.project_name, f'{sequence:02d}', 'info.h5')
            with h5py.File(info_file, 'r+') as f:
                f['selection_mask'][:mask.shape[0]] = mask

    def label_voxels(self, voxels: np.ndarray, cloud_path: str) -> None:
        """ Label the voxels in the dataset. This is used when the dataset is used in an active training when
        VoxelSelector object selects the samples to label. It updates the selection mask and the label masks
        which contains points inside the selected voxels.

        :param voxels: Array of voxel indices. (N,)
        :param cloud_path: Path to the cloud that contains the selected voxels.
        """

        labels = self.label_files[np.where(self.cloud_map == cloud_path)[0]]
        sample_map = self.sample_map[np.where(self.cloud_map == cloud_path)[0]]

        # Iterate over all labels, which points can be inside the selected voxels
        for label_file, sample_idx in tqdm(zip(labels, sample_map), total=len(labels), desc='Labeling voxels'):

            # Update the label mask on places where the points are inside the selected voxels
            with h5py.File(label_file, 'r') as f:
                voxel_map = np.asarray(f['voxel_map'])
            with h5py.File(label_file.replace('sequences', self.project_name), 'r+') as f:
                label_mask = f['label_mask']
                label_mask[np.isin(voxel_map, voxels)] = 1

                # If there are any labeled points in the label mask, update the selection mask for the sample
                if np.sum(label_mask) > 0:
                    self.selection_mask[sample_idx] = 1

        # Update the selection masks also on the disk
        self.update_sequence_selection_masks()

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
        # Load scan
        with h5py.File(self.scans[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions']).flatten()

        # Load label
        with h5py.File(self.labels[idx], 'r') as f:
            labels = np.asarray(f['labels']).flatten()
            voxel_map = np.asarray(f['voxel_map']).flatten().astype(np.float32)
            labels = map_labels(labels, self.label_map)

        if self.selection_mode:
            voxel_map[labels == self.ignore_index] = -1
        elif self.split == 'train':
            # Apply label mask
            with h5py.File(self.labels[idx].replace('sequences', self.project_name), 'r') as f:
                labels *= np.asarray(f['label_mask']).flatten()

            # Apply augmentations
            points, drop_mask = augment_points(points,
                                               drop_prob=0.5,
                                               flip_prob=0.5,
                                               rotation_prob=0.5,
                                               translation_prob=0.5)
            labels = labels[drop_mask]
            colors = colors[drop_mask]
            remissions = remissions[drop_mask]

        # Project points to image and map the projection
        proj = project_points(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_distances, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        # Project remissions
        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        # Project colors
        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        # Project labels
        proj_labels = np.zeros((self.proj_H, self.proj_W), dtype=np.long)
        proj_labels[proj_mask] = labels[proj_idx[proj_mask]]

        # Project voxel map
        proj_voxel_map = np.full((self.proj_H, self.proj_W), -1, dtype=np.int64)
        proj_voxel_map[proj_mask] = voxel_map[proj_idx[proj_mask]]

        # Concatenate scan features
        proj_scan = np.concatenate([proj_distances[..., np.newaxis],
                                    proj_remissions[..., np.newaxis],
                                    proj_colors], axis=-1, dtype=np.float32).transpose((2, 0, 1))

        return proj_scan, proj_labels, proj_voxel_map, self.get_cloud_id(idx), self.end_of_voxel_cloud(idx)

    def get_most_labeled_sample(self) -> tuple[int, float, np.ndarray]:
        max_label_ratio, most_labeled_sample = 0, 0
        sample_label_mask = np.zeros(self.__len__(), dtype=np.float32)

        for i in range(self.__len__()):
            with h5py.File(self.label_files[i].replace('sequences', self.project_name), 'r') as f:
                label_mask = np.asarray(f['label_mask']).flatten()
                label_ratio = np.sum(label_mask) / len(label_mask)
                if label_ratio > max_label_ratio:
                    max_label_ratio = label_ratio
                    most_labeled_sample = i
                    sample_label_mask = label_mask

        return most_labeled_sample, max_label_ratio, sample_label_mask

    def get_statistics(self, ignore: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """ Returns the statistics of the dataset. The statistics are the class distribution, the labeling
        progress for each class and the labeling progress for the dataset.

        :param ignore: The label to ignore when calculating the statistics
        :return: The class distribution, the labeling progress for each class and the labeling progress for the dataset
        """

        counter = 0
        labeled_counter = 0

        class_counts = np.zeros(self.num_classes, dtype=np.long)
        labeled_class_counts = np.zeros(self.num_classes, dtype=np.long)

        for path in tqdm(self.voxel_clouds, desc='Calculating dataset statistics'):
            with h5py.File(path, 'r') as f:
                labels = np.asarray(f['labels']).flatten()
                labels = map_labels(labels, self.label_map)

            with h5py.File(path.replace('sequences', self.project_name), 'r') as f:
                label_mask = np.asarray(f['label_mask']).flatten()
                voxel_mask = self.get_voxel_mask(path, len(labels))
                labels *= voxel_mask

                # Add counts to the class counts
                all_labels = labels[labels != ignore]
                unique_labels, counts = np.unique(all_labels, return_counts=True)
                class_counts[unique_labels] += counts
                counter += np.sum(counts)

                # Add counts to the labeled class counts
                selected_labels = labels * label_mask
                selected_labels = selected_labels[selected_labels != ignore]
                unique_labels, counts = np.unique(selected_labels, return_counts=True)
                labeled_class_counts[unique_labels] += counts
                labeled_counter += np.sum(counts)

        # Calculate the class distribution in the whole dataset
        class_distribution = class_counts / (counter + 1e-6)

        # Calculate the class distribution in the labeled dataset
        labeled_class_distribution = labeled_class_counts / (labeled_counter + 1e-6)

        # Calculate the labeling progress for each class
        class_progress = labeled_class_counts / (class_counts + 1e-6)

        # Calculate the labeling progress for the whole dataset
        dataset_labeled_ratio = labeled_counter / (counter + 1e-6)

        return class_distribution, labeled_class_distribution, class_progress, dataset_labeled_ratio

    def get_voxel_mask(self, cloud_path: str, cloud_size: int) -> np.ndarray:
        voxel_mask = np.zeros(cloud_size, dtype=np.bool)
        sample_indices = self.sample_map[np.where(self.cloud_map == cloud_path)[0]]

        for i in sample_indices:
            with h5py.File(self.label_files[i], 'r') as f:
                voxel_map = np.asarray(f['voxel_map']).flatten()
                voxel_map = voxel_map[(voxel_map != -1)]
                voxel_mask[voxel_map] = True

        return voxel_mask

    def __len__(self):
        return len(self.scans)

    def __str__(self):
        return f'\nSemanticDataset: {self.split}\n' \
               f'\t - Dataset size: {self.__len__()} / {len(self.scan_files)}\n' \
               f'\t - Sequences: {self.sequences}\n'

    def __repr__(self):
        return self.__str__()
