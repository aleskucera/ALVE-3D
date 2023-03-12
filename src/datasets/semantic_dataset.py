import h5py
import torch
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils import project_points, map_labels, augment_points
from .utils import load_sample_selection_mask, load_semantic_dataset


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

    def __init__(self, dataset_path: str, cfg: DictConfig, split: str, size: int = None,
                 active_mode: bool = False, sequences: iter = None):

        assert split in ['train', 'val']

        self.cfg = cfg
        self.size = size
        self.split = split
        self.path = dataset_path
        self.label_map = cfg.learning_map
        self.active = active_mode

        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = cfg.split[split]

        self.proj_W = cfg.projection.W
        self.proj_H = cfg.projection.H
        self.proj_fov_up = cfg.projection.fov_up
        self.proj_fov_down = cfg.projection.fov_down

        self.poses = None
        self.scans = None
        self.labels = None

        self.cloud_map = None
        self.sample_map = None
        self.sequence_map = None
        self.selection_mask = None

        self._initialize()

    @property
    def scan_files(self):
        """ Returns an array of scans that are currently
        selected. (labels are available)
        """

        selected = np.where(self.selection_mask == 1)[0]
        return self.scans[selected]

    @property
    def label_files(self):
        """ Returns an array of labels that are currently
        selected. (labels are available)
        """

        selected = np.where(self.selection_mask == 1)[0]
        return self.labels[selected]

    def _initialize(self):
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

        data = load_semantic_dataset(self.path, self.sequences, self.split, self.active)
        self.scans, self.labels, self.poses, self.sequence_map, self.cloud_map, self.selection_mask = data

        self.poses = self.poses[:self.size]
        self.scans = self.scans[:self.size]
        self.labels = self.labels[:self.size]
        self.sequence_map = self.sequence_map[:self.size]
        self.cloud_map = self.cloud_map[:self.size]
        self.selection_mask = self.selection_mask[:self.size]

        self.sample_map = np.arange(len(self.scans))

    def update(self):
        """Loads the sample selection mask of the dataset from a disk. This is used when the dataset is
        used in an active training loop after the new labels are available.
        """

        self.selection_mask = load_sample_selection_mask(self.path, self.sequences, self.split)[:self.size]

    def label_voxels(self, voxels: np.ndarray, sequence: int, cloud: int):
        """ Label the voxels in the dataset. This is used when the dataset is used in an active training when
        VoxelSelector object selects the samples to label. It updates the selection mask and the label masks
        which contains points inside the selected voxels.

        :param voxels: Array of voxel indices. (N,)
        :param sequence: Sequence index.
        :param cloud: Cloud index relative to the sequence.
        """

        # Select labels, cloud indices and sample indices for a given sequence
        seq_labels = self.labels[np.where(self.sequence_map == sequence)[0]]
        seq_cloud_map = self.cloud_map[np.where(self.sequence_map == sequence)[0]]
        seq_sample_map = self.sample_map[np.where(self.sequence_map == sequence)[0]]

        # From selected labels select the ones that belong to a given cloud
        labels = seq_labels[np.where(seq_cloud_map == cloud)[0]]
        sample_map = seq_sample_map[np.where(seq_cloud_map == cloud)[0]]

        for label_file, sample_idx in zip(labels, sample_map):

            # Update the label mask
            with h5py.File(label_file, 'r+') as f:
                voxel_map = f['voxel_map']
                label_mask = f['label_mask']
                label_mask[np.isin(voxel_map, voxels)] = 1

                # Update the selection mask
                if np.sum(label_mask) > 0:
                    self.selection_mask[sample_idx] = 1

    def label_samples(self, sample_indices: np.ndarray):
        """Label the samples in the dataset. This is used when the dataset is used in an active training when
        SampleSelector object selects the samples to label. It updates the selection mask and the label masks
        of the specified samples.

        :param sample_indices: Array of sample indices. (N,)
        """

        # Update the selection mask
        self.selection_mask[sample_indices] = 1

        # Update the label masks
        for idx in sample_indices:
            with h5py.File(self.labels[idx], 'r+') as f:
                label_mask = f['label_mask']
                label_mask[:] = 1

    def get_item(self, idx) -> tuple[np.ndarray, np.ndarray, int, int]:
        """ Return a sample from the dataset. Typically used in an active learning loop for selecting samples
        to label. The difference between this function and __getitem__ is that this function returns the following data:

        - Projected scan to the 2D plane (5, H, W)
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
            voxel_map = np.asarray(f['voxel_map']).flatten()

        # Project points to image
        proj = project_points(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_depth, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        proj_image = np.concatenate([proj_depth[..., np.newaxis],
                                     proj_remissions[..., np.newaxis],
                                     proj_colors], axis=-1, dtype=np.float32)

        proj_voxel_map = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        proj_voxel_map[proj_mask] = voxel_map[proj_idx[proj_mask]]

        proj_image = proj_image.transpose((2, 0, 1))
        return proj_image, proj_voxel_map, self.sequence_map[idx], self.cloud_map[idx]

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
            label_mask = np.asarray(f['label_mask']).flatten()

            # Map labels to training labels
            labels = map_labels(labels, self.label_map)

        if self.split == 'train':
            # Apply label mask
            labels *= label_mask

            # Apply augmentations
            points, drop_mask = augment_points(points, translation_prob=0.5,
                                               rotation_prob=0.5, flip_prob=0.5,
                                               drop_prob=0.5)
            labels = labels[drop_mask]
            colors = colors[drop_mask]
            remissions = remissions[drop_mask]

        # Project points to image and map the projection
        proj = project_points(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_depth, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        # Project remissions
        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        # Project colors
        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        # Project labels
        proj_labels = np.zeros((self.proj_H, self.proj_W), dtype=np.long)
        proj_labels[proj_mask] = labels[proj_idx[proj_mask]]

        # Concatenate the projected image
        proj_image = np.concatenate([proj_depth[..., np.newaxis],
                                     proj_remissions[..., np.newaxis],
                                     proj_colors], axis=-1, dtype=np.float32)

        proj_image = proj_image.transpose((2, 0, 1))

        return proj_image, proj_labels

    def __len__(self):
        return np.sum(self.selection_mask)

    def get_length(self):
        return len(self.scans)

    def get_dataset_structure(self):
        order = np.argsort(self.sequence_map)
        sequence_map = self.sequence_map[order]
        cloud_map = self.cloud_map[order]

        clouds_ids = np.array([], dtype=np.long)
        sequences = np.array([], dtype=np.long)
        for seq in np.unique(sequence_map):
            seq_cloud_ids = np.unique(cloud_map[sequence_map == seq])
            clouds_ids = np.concatenate((clouds_ids, seq_cloud_ids))
            sequences = np.concatenate((sequences, np.full_like(seq_cloud_ids, seq)))

        return clouds_ids, sequences

    def __str__(self):
        return f'\nSemanticDataset: {self.split}\n' \
               f'\t - Dataset size: {len(self)}\n' \
               f'\t - Sequences: {self.sequences}\n'
