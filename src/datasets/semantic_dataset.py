import os
import logging

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils import project_points, map_labels, augment_points
from .utils import load_semantic_dataset

log = logging.getLogger(__name__)


class SemanticDataset(Dataset):
    """PyTorch dataset wrapper for the semantic segmentation dataset. This dataset is used also for active learning.


    :param dataset_path: Path to the dataset.
    :param cfg: Configuration of the dataset.
    :param split: Split to load. ('train', 'val')
    :param size: Number of samples to load. If None all the samples are loaded (default: None)
    :param active_mode: If True the dataset initializes in active
                                 mode - deactivates all labels. (default: False)
    :param sequences: List of sequences to load. If None all the sequences defined in config
                      for a given split are loaded. (default: None)
    """

    def __init__(self, dataset_path: str, project_name: str, cfg: DictConfig, split: str, size: int = None,
                 active_mode: bool = False, sequences: iter = None, resume: bool = False):

        assert split in ['train', 'val']

        self.cfg = cfg
        self.size = size
        self.resume = resume
        self.split = split
        self.path = dataset_path
        self.active = active_mode
        self.project_name = project_name
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

        self.scans = None
        self.labels = None

        self.cloud_map = None
        self.sample_map = None
        self.sequence_map = None
        self.selection_mask = None

        self._initialize()

    @property
    def scan_files(self) -> np.ndarray:
        """ Returns an array of scans that are currently
        selected. (labels are available)
        """

        selected = np.where(self.selection_mask == 1)[0]
        return self.scans[selected]

    @property
    def label_files(self) -> np.ndarray:
        """ Returns an array of labels that are currently
        selected. (labels are available)
        """

        selected = np.where(self.selection_mask == 1)[0]
        return self.labels[selected]

    def _initialize(self) -> None:
        """ Initialize the dataset. Sets the sample selection masks and sample masks to 0 or 1 based on the
        active learning mode. Then loads the following data:

        - scans: Array of paths to scans. (N,)
        - labels: Array of paths to labels. (N,)
        - poses: Array of poses. (N, 4, 4)
        - sequence_map: Array of sequence indices that maps each sample to a sequence. (N,)
        - cloud_map: Array of cloud indices that maps each sample to a global cloud. (N,)
        - selection_mask: Array of 0 or 1 that indicates if the sample is selected or not. (N,)

        Then the function crops the dataset to the desired size (self.size) and creates a sample map that maps
        the samples to the original dataset.
        """

        data = load_semantic_dataset(self.path, self.project_name, self.sequences, self.split, self.active, self.resume)
        self.scans, self.labels, self.sequence_map, self.cloud_map, self.selection_mask = data

        self.scans = self.scans[:self.size]
        self.labels = self.labels[:self.size]
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

        labels = self.labels[np.where(self.cloud_map == cloud_path)[0]]
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

    def label_samples(self, sample_indices: np.ndarray) -> None:
        """Label the samples in the dataset. This is used when the dataset is used in an active training when
        SampleSelector object selects the samples to label. It updates the selection mask and the label masks
        of the specified samples.

        :param sample_indices: Array of sample indices. (N,)
        """

        # Update the selection mask
        self.selection_mask[sample_indices] = 1

        # Update the label masks
        for idx in tqdm(sample_indices, desc='Updating label masks'):
            with h5py.File(self.labels[idx].replace('sequences', self.project_name), 'r+') as f:
                label_mask = f['label_mask']
                label_mask[...] = np.ones_like(f['label_mask'])

        # Update the selection masks also on the disk
        self.update_sequence_selection_masks()

    def get_item(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """ Return a sample from the dataset. Typically used in an active learning loop for
        selecting samples to label. The difference between this function and __getitem__ is that
        this function returns the following data:

        - Projected scan to the 2D plane (5, H, W)
        - Projected radial distances to the 2D plane (H, W)
        - Projected voxel map to the 2D plane (H, W)
        - Sequence index
        - Cloud index

        This function does not apply augmentations and does not return labels.

        :param idx: Sample index.
        :return: Projected scan, projected voxel map, sequence index, cloud index.
        """

        # Load scan
        with h5py.File(self.scans[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions']).flatten()

        # Load voxel map
        with h5py.File(self.labels[idx], 'r') as f:
            labels = np.asarray(f['labels']).flatten()
            voxel_map = np.asarray(f['voxel_map']).flatten().astype(np.float32)
            labels = map_labels(labels, self.label_map)
            voxel_map[labels == self.ignore_index] = -1

        # Project points to image and map the projection
        proj = project_points(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_distances, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        # Project remissions
        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        # Project colors
        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        # Concatenate scan features
        proj_scan = np.concatenate([proj_distances[..., np.newaxis],
                                    proj_remissions[..., np.newaxis],
                                    proj_colors], axis=-1, dtype=np.float32).transpose((2, 0, 1))

        # Project voxel map
        proj_voxel_map = np.full((self.proj_H, self.proj_W), -1, dtype=np.int64)
        proj_voxel_map[proj_mask] = voxel_map[proj_idx[proj_mask]]

        return proj_scan, proj_distances, proj_voxel_map, self.cloud_map[idx]

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """Returns a sample from the dataset. The sample is projected velodyne scan and the corresponding
        label.

        The function creates a sample by following the steps:
        1. Load the scan and the label from the disk.
        2. Apply the label mask to the label - some labels may be hidden for
           training. (Only for active learning mode and training split)
        3. Apply the augmentations to the scan and the label. (Only for training split)
        4. Map the labels to the training labels.
        5. Project the scan to the image.

        :param idx: Sample index (indexing only from the labeled samples)
        :return: Projected scan, projected label
        """

        # Load scan
        with h5py.File(self.scan_files[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions']).flatten()

        # Load label
        with h5py.File(self.label_files[idx], 'r') as f:
            labels = np.asarray(f['labels']).flatten()
            labels = map_labels(labels, self.label_map)

        with h5py.File(self.label_files[idx].replace('sequences', self.project_name), 'r') as f:
            label_mask = np.asarray(f['label_mask']).flatten()

        if self.split == 'train':
            # Apply label mask
            labels *= label_mask

            # Apply augmentations
            points, drop_mask = augment_points(points, translation_prob=0.5,
                                               rotation_prob=0.5, flip_prob=0.5,
                                               drop_prob=0.5)
            labels, colors, remissions = labels[drop_mask], colors[drop_mask], remissions[drop_mask]

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

        # Concatenate scan features
        proj_image = np.concatenate([proj_distances[..., np.newaxis],
                                     proj_remissions[..., np.newaxis],
                                     proj_colors], axis=-1, dtype=np.float32).transpose((2, 0, 1))

        return proj_image, proj_labels

    def __len__(self):
        return np.sum(self.selection_mask)

    def get_full_length(self):
        return len(self.scans)

    def get_dataset_clouds(self) -> np.ndarray:
        return np.unique(self.cloud_map)

    def get_most_labeled_sample(self) -> tuple[int, float, np.ndarray]:

        max_label_ratio = 0
        most_labeled_sample = 0
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

        cloud_paths = self.get_dataset_clouds()

        counter = 0
        labeled_counter = 0

        class_counts = np.zeros(self.num_classes, dtype=np.long)
        labeled_class_counts = np.zeros(self.num_classes, dtype=np.long)

        for path in tqdm(cloud_paths, desc='Calculating dataset statistics'):
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
        sample_map = self.sample_map[np.where(self.cloud_map == cloud_path)[0]]

        # Iterate over all labels, which points can be inside the selected voxels
        for i in sample_map:
            _, _, voxel_map, _ = self.get_item(i)
            voxel_map = voxel_map.flatten()
            valid = (voxel_map != -1)
            voxel_map = voxel_map[valid]
            voxel_mask[voxel_map] = True

        return voxel_mask

    def __str__(self):
        return f'\nSemanticDataset: {self.split}\n' \
               f'\t - Dataset size: {self.__len__()} / {self.get_full_length()}\n' \
               f'\t - Sequences: {self.sequences}\n'

    def __repr__(self):
        return self.__str__()
