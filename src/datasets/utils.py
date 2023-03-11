import os

import h5py
import numpy as np


def initialize_semantic_samples(dataset_path: str, sequences: list, active: bool = False) -> None:
    """ Initialize semantic samples for a given split and mode. The initialization is done by setting the selection mask
    and the label mask to 0 or 1 depending on the mode.

    If active is True, the train selection mask and the label mask are set to 0. This means that the samples are not selected
    and the labels are not available.

    If active is False, the train selection mask and the label mask are set to 1. This means that the samples are selected
    and the labels are available. (default)

    Validation samples are always selected and the labels are available.

    :param dataset_path: Path to the dataset.
    :param sequences: List of sequences to initialize.
    :param active: If True, the samples are initialized as not selected and the labels are not available.
    """

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]

    for sequence_dir in sequence_dirs:
        info_path = os.path.join(sequence_dir, 'info.h5')
        labels_dir = os.path.join(sequence_dir, 'labels')

        with h5py.File(info_path, 'r+') as f:
            train_indices = np.asarray(f['train'])
            if active:
                f['selection_mask'][...] = np.zeros_like(f['selection_mask'])
            else:
                f['selection_mask'][...] = np.ones_like(f['selection_mask'])

        # Get the train labels for the given sequence
        seq_labels = sorted([os.path.join(labels_dir, l) for l in os.listdir(labels_dir) if l.endswith('.h5')])
        split_seq_labels = np.array(seq_labels, dtype=np.str_)[train_indices]

        # Open the label files and set the label mask to 0 or 1
        for label_path in split_seq_labels:
            with h5py.File(label_path, 'r+') as f:
                if active:
                    f['label_mask'][...] = np.zeros_like(f['label_mask'])
                else:
                    f['label_mask'][...] = np.ones_like(f['label_mask'])


def load_semantic_dataset(dataset_path: str, sequences: list, split: str, active: bool = False) -> tuple:
    """ Load the semantic dataset for a given split and mode. The function loads the following information:

    - scans: Array of paths to the scans.
    - labels: Array of paths to the labels.
    - poses: Array of poses.
    - sequence_map: Array of sequence indices.
    - cloud_map: Array of cloud indices.
    - selection_mask: Array of selection masks.

    :param dataset_path: Path to the dataset.
    :param sequences: List of sequences to load.
    :param split: Split to load.
    :param active: If True, the samples are loaded as not selected and the labels are not available.
    :return: Tuple containing the scans, labels, poses, sequence map, cloud map, and selection mask.
    """

    initialize_semantic_samples(dataset_path, sequences, active)

    scans = np.array([], dtype=np.str_)
    labels = np.array([], dtype=np.str_)
    poses = np.array([], dtype=np.float32).reshape((0, 4, 4))

    cloud_map = np.array([], dtype=np.int32)
    sequence_map = np.array([], dtype=np.int32)

    selection_mask = np.array([], dtype=bool)

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]

    for sequence, sequence_dir in zip(sequences, sequence_dirs):
        info_path = os.path.join(sequence_dir, 'info.h5')
        labels_dir = os.path.join(sequence_dir, 'labels')
        scans_dir = os.path.join(sequence_dir, 'velodyne')

        with h5py.File(info_path, 'r+') as f:
            split_indices = np.asarray(f[split])
            seq_poses = np.asarray(f['poses'])[split_indices]
            seq_cloud_map = np.asarray(f['cloud_map'])[split_indices]

            if split == 'train':
                seq_selection_mask = np.asarray(f['selection_mask'])
            elif split == 'val':
                seq_selection_mask = np.ones_like(split_indices)

            poses = np.concatenate((poses, seq_poses), axis=0)
            cloud_map = np.concatenate((cloud_map, seq_cloud_map), axis=0)
            selection_mask = np.concatenate((selection_mask, seq_selection_mask), axis=0)

        seq_scans = sorted([os.path.join(scans_dir, v) for v in os.listdir(scans_dir) if v.endswith('.h5')])
        seq_labels = sorted([os.path.join(labels_dir, l) for l in os.listdir(labels_dir) if l.endswith('.h5')])

        seq_split_scans = np.array(seq_scans, dtype=np.str_)[split_indices]
        seq_split_labels = np.array(seq_labels, dtype=np.str_)[split_indices]

        scans = np.concatenate((scans, seq_split_scans), axis=0)
        labels = np.concatenate((labels, seq_split_labels), axis=0)
        sequence_map = np.concatenate((sequence_map, np.full_like(split_indices, sequence)), axis=0)

    return scans, labels, poses, sequence_map, cloud_map, selection_mask


def load_sample_selection_mask(dataset_path: str, sequences: list, split: str) -> np.ndarray:
    """ Load the sample selection mask for a given split.

    :param dataset_path: Path to the dataset.
    :param sequences: List of sequences to load.
    :param split: Split to load.
    :return: Array of selection mask.
    """

    assert split in ['train', 'val']

    selection_mask = np.array([], dtype=bool)

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]

    for sequence_dir in sequence_dirs:
        info_path = os.path.join(sequence_dir, 'info.h5')
        with h5py.File(info_path, 'r') as f:
            if split == 'train':
                seq_selection_mask = np.asarray(f['selection_mask'])
                selection_mask = np.concatenate((selection_mask, seq_selection_mask), axis=0)
            elif split == 'val':
                selection_mask = np.concatenate((selection_mask, np.ones_like(f[split])), axis=0)
    return selection_mask
