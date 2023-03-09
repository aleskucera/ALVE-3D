import h5py
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.project import project_scan
from .utils import update_selection_mask, load_semantic_dataset, crop_sequence_format


class SemanticDataset(Dataset):
    """Semantic dataset for active learning.

    :param dataset_path: Path to the dataset.
    :param cfg: Configuration object.
    :param split: Split to load. ('train', 'val')
    :param size: Number of samples to load. If None all the samples are loaded (default: None)
    """

    def __init__(self, dataset_path: str, cfg: DictConfig, split: str, mode: str = 'passive', sequences: iter = None,
                 size: int = None):

        assert split in ['train', 'val']
        assert mode in ['passive', 'active']

        self.cfg = cfg
        self.mode = mode
        self.size = size
        self.split = split
        self.path = dataset_path

        if sequences is None:
            self.sequences = cfg.split[split]
        else:
            self.sequences = sequences

        self.proj_W = cfg.projection.W
        self.proj_H = cfg.projection.H
        self.proj_fov_up = cfg.projection.fov_up
        self.proj_fov_down = cfg.projection.fov_down

        self._poses = None
        self._scans = None
        self._labels = None

        self.cloud_maps = None
        self.selection_masks = None

        self._initialize()

    @property
    def scan_files(self):
        scans = np.concatenate(self._scans)

        selection_mask = np.concatenate(self.selection_masks)
        indices = np.where(selection_mask == 1)[0]

        scans = scans[indices]
        return scans

    @property
    def label_files(self):
        labels = np.concatenate(self._labels)

        selection_mask = np.concatenate(self.selection_masks)
        indices = np.where(selection_mask == 1)[0]

        labels = labels[indices]
        return labels

    @property
    def poses(self):
        poses = np.concatenate(self._poses)

        selection_mask = np.concatenate(self.selection_masks)
        indices = np.where(selection_mask == 1)[0]

        poses = poses[indices]
        return poses

    def _initialize(self):
        """Initialize the dataset. Load the scans, labels, poses and selection masks. The data has following shape:

        scans: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence
        labels: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence
        poses: (S, N_i, 4, 4) - S: number of sequences, N: number of samples in the i-th sequence
        cloud_maps: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence
        selection_masks: (S, N_i) - S: number of sequences, N: number of samples in the i-th sequence

        After that the dataset is cropped to the specified size so that sum(N_i for i = 1, ... , S) = size.
        """

        data = load_semantic_dataset(self.path, self.sequences, self.split, self.mode)
        self._scans, self._labels, self._poses, self.cloud_maps, self.selection_masks = data

        # Crop the dataset if the size is specified
        if self.size is not None:
            crop_sequence_format(self._poses, self.size)
            crop_sequence_format(self._scans, self.size)
            crop_sequence_format(self._labels, self.size)
            crop_sequence_format(self.cloud_maps, self.size)
            crop_sequence_format(self.selection_masks, self.size)

    def update(self):
        """Update the selection masks. This is used when the dataset is used in an active
        training loop after the new labels are available.
        """

        self.selection_masks = update_selection_mask(self.path, self.sequences, self.split)

        if self.size is not None:
            crop_sequence_format(self.selection_masks, self.size)

    def label_global_voxels(self, voxels: np.ndarray, sequence: int):
        """Select the labels for training based on the global voxel indices.

        :param voxels: (C, V) - C: number of clouds, V: number of selected voxels in the i-th cloud
        :param sequence: The sequence index (index is relative to the split)
        """

        assert sequence < len(self.sequences), f'Invalid sequence index: {sequence}'

        seq_labels = self._labels[sequence]
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

        seq_labels = self._labels[sequence]
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

        # Load scan
        with h5py.File(self.scan_files[idx], 'r') as f:
            points = np.asarray(f['points'])
            colors = np.asarray(f['colors'])
            remissions = np.asarray(f['remissions']).flatten()

        # Load label
        with h5py.File(self.label_files[idx], 'r') as f:
            labels = np.asarray(f['labels']).flatten()
            label_mask = np.asarray(f['label_mask']).flatten()

        # Apply mask
        labels *= label_mask

        # Project points to image
        proj = project_scan(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_depth, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

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
