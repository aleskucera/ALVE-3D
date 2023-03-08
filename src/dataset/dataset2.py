import h5py
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.project import project_scan
from src.utils.io import update_selection_mask, load_dataset


class ActiveDataset(Dataset):
    """Semantic dataset for active learning.

    :param dataset_path: Path to the dataset.
    :param cfg: Configuration object.
    :param split: Split to load. ('train', 'val')
    :param size: Number of samples to load. If None all the samples are loaded (default: None)
    """

    def __init__(self, dataset_path: str, cfg: DictConfig, split: str, size: int = None):
        self.cfg = cfg
        self.size = size
        self.split = split
        self.path = dataset_path
        self.sequences = cfg.split[split]

        self.proj_W = cfg.projection.W
        self.proj_H = cfg.projection.H
        self.proj_fov_up = cfg.projection.fov_up
        self.proj_fov_down = cfg.projection.fov_down

        self.poses = None
        self.scans = None
        self.labels = None
        self.cloud_maps = None
        self.selection_masks = None

        self._init()

    def _init(self):
        """Initialize the dataset. Load the scans, labels, poses and selection masks. The data has following shape:

        scans: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence
        labels: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence
        poses: (S, N_i, 4, 4) - S: number of sequences, N: number of samples in the i-th sequence
        cloud_maps: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence
        selection_masks: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence

        After that the dataset is cropped to the specified size so that sum(N_i for i = 1, ... , S) = size.
        """

        self.scans, self.labels, self.poses, self.selection_masks = load_dataset(self.path, self.sequences, self.split)

        if self.size is not None:
            self._crop_dataset(self.size)

    def _crop_dataset(self, size: int):
        """Crop the dataset to the specified size so that sum(N_i for i = 1, ... , S) = size, where
        N_i is the number of samples in the i-th sequence and S is the number of sequences.
        """

        # Compute the sequence index where the size is located
        seq_idx = 0
        seq_size = len(self.scans[seq_idx])
        while seq_size < size:
            seq_idx += 1
            seq_size += len(self.scans[seq_idx])

        # Compute the sample index where to crop the sequence
        sample_idx = size - seq_size + len(self.scans[seq_idx])

        # Select the whole sequences
        self.cropped_poses = self.poses[:seq_idx]
        self.cropped_scans = self.scans[:seq_idx]
        self.cropped_labels = self.labels[:seq_idx]
        self.cropped_selection_mask = self.selection_masks[:seq_idx]

        # Append the cropped sequence
        self.cropped_poses.append(self.poses[seq_idx][:sample_idx])
        self.cropped_scans.append(self.scans[seq_idx][:sample_idx])
        self.cropped_labels.append(self.labels[seq_idx][:sample_idx])
        self.cropped_selection_mask.append(self.selection_masks[seq_idx][:sample_idx])

    def update(self):
        """Update the selection masks. This is used when the dataset is used in an active
        training loop after the new labels are available.
        """

        self.selection_masks = update_selection_mask(self.path, self.sequences, self.split)

    def label_global_voxels(self, voxels: np.ndarray, sequence: int):
        """Select the labels for training based on the global voxel indices.

        :param voxels: (C, V) - C: number of clouds, V: number of selected voxels in the i-th cloud
        :param sequence: The sequence index (index is relative to the split)
        """

        assert sequence < len(self.sequences), f'Invalid sequence index: {sequence}'

        seq_labels = self.labels[sequence]
        seq_cloud_map = self.cloud_maps[sequence]
        seq_selection_mask = self.selection_masks[sequence]

        for cloud_idx, voxels in enumerate(voxels):

            # Get the sample indices that are in the cloud
            sample_indices = np.where(seq_cloud_map == cloud_idx)[0]
            for sample_idx in sample_indices:

                # Load the voxel map and current label mask
                with h5py.File(seq_labels[sample_idx], 'r+') as f:
                    voxel_map = f['voxel_map']
                    label_mask = f['label_mask']

                    # Set the label mask to 1 for selected voxels and update the selection mask
                    label_mask[np.isin(voxel_map, voxels)] = 1
                    if np.sum(label_mask) > 0:
                        seq_selection_mask[sample_idx] = 1

    def label_whole_samples(self, sample_indices: np.ndarray, sequence: int):
        """ Select the labels for training based on the sample indices.

        :param sample_indices: (N) - N: number of selected samples
        :param sequence: The sequence index (index is relative to the split)
        """

        assert sequence < len(self.sequences), f'Invalid sequence index: {sequence}'

        seq_labels = self.labels[sequence]
        seq_selection_mask = self.selection_masks[sequence]
        seq_selection_mask[sample_indices] = 1

        for sample_idx in sample_indices:
            with h5py.File(seq_labels[sample_idx], 'r+') as f:
                label_mask = f['label_mask']
                label_mask[:] = 1

    def __getitem__(self, idx):
        """ Return the idx-th sample in the dataset which is selected by the selection mask.
        The sample is a tuple of the projected image and the projected labels.
        """

        # Flatten the samples
        scans = np.concatenate(self.scans)
        labels = np.concatenate(self.labels)

        selection_mask = np.concatenate(self.selection_masks)
        indices = np.where(selection_mask == 1)[0]

        scans = scans[indices]
        labels = labels[indices]

        # Load scan
        with h5py.File(scans[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions'])

        # Load label
        with h5py.File(labels[idx], 'r') as f:
            labels = np.asarray(f['labels'])
            label_mask = np.asarray(f['label_mask'])

        # Apply mask
        labels *= label_mask

        # Project points to image
        p = project_scan(points, self.proj_W, self.proj_H, self.proj_fov_up, self.proj_fov_down)
        proj_depth, proj_idx, proj_mask = p['proj_depth'], p['proj_idx'], p['proj_mask']

        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

        proj_labels = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        proj_labels[proj_mask] = labels[proj_idx[proj_mask]]

        proj_image = np.concatenate([proj_depth[..., np.newaxis],
                                     proj_remissions[..., np.newaxis],
                                     proj_colors], axis=-1)

        return proj_image, proj_labels

    def __len__(self):
        """Return the number of samples in the dataset.
        This is the sum of the number of samples in each sequence.
        """

        total_samples = 0
        for seq_mask in self.selection_masks:
            total_samples += np.sum(seq_mask)
        return total_samples

    def __str__(self):
        return f'\nSemanticDataset: {self.split}\n' \
               f'\t - Dataset size: {len(self)}\n' \
               f'\t - Sequences: {self.sequences}\n'
