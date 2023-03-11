import os

import h5py
import numpy as np


def initialize_semantic_samples(dataset_path: str, sequences: list, split: str, mode: str) -> None:
    assert mode in ['passive', 'active']
    for sequence in sequences:
        sequence_path = os.path.join(dataset_path, 'sequences', f'{sequence:02d}')

        info_path = os.path.join(sequence_path, 'info.h5')
        labels_path = os.path.join(os.path.join(sequence_path, 'labels'))

        with h5py.File(info_path, 'r+') as f:
            split_indices = np.asarray(f[split])
            if mode == 'active':
                f['selection_mask'][...] = np.zeros_like(f['selection_mask'])
            elif mode == 'passive':
                f['selection_mask'][...] = np.ones_like(f['selection_mask'])
            else:
                raise ValueError(f'Invalid mode: {mode}')

        split_labels = [os.path.join(labels_path, l) for l in os.listdir(labels_path) if l.endswith('.h5')]
        split_labels.sort()
        split_labels = np.array(split_labels, dtype=np.str_)[split_indices]

        for label in split_labels:
            label_path = os.path.join(labels_path, label)
            with h5py.File(label_path, 'r+') as f:
                if mode == 'active':
                    f['label_mask'][...] = np.zeros_like(f['label_mask'])
                elif mode == 'passive':
                    f['label_mask'][...] = np.ones_like(f['label_mask'])
                else:
                    raise ValueError(f'Invalid mode: {mode}')


def load_semantic_dataset(dataset_path: str, sequences: list, split: str, dataset_mode: str) -> tuple:
    assert split in ['train', 'val']

    initialize_semantic_samples(dataset_path, sequences, split=split, mode=dataset_mode)

    poses = []
    labels = []
    velodyne = []
    cloud_maps = []
    selection_masks = []

    for sequence in sequences:
        sequence_path = os.path.join(dataset_path, 'sequences', f'{sequence:02d}')

        info_path = os.path.join(sequence_path, 'info.h5')
        labels_path = os.path.join(os.path.join(sequence_path, 'labels'))
        velodyne_path = os.path.join(os.path.join(sequence_path, 'velodyne'))

        with h5py.File(info_path, 'r+') as f:
            split_indices = np.asarray(f[split])

            poses.append(np.asarray(f['poses'])[split_indices])
            cloud_maps.append(np.asarray(f['cloud_map'])[split_indices])

            if split == 'train':
                selection_masks.append(np.asarray(f['selection_mask']))
            elif split == 'val':
                selection_masks.append(np.ones_like(split_indices))

        seq_labels = [os.path.join(labels_path, l) for l in os.listdir(labels_path) if l.endswith('.h5')]
        seq_velodyne = [os.path.join(velodyne_path, v) for v in os.listdir(velodyne_path) if v.endswith('.h5')]

        seq_labels.sort()
        seq_velodyne.sort()

        labels.append(np.array(seq_labels, dtype=np.str_)[split_indices])
        velodyne.append(np.array(seq_velodyne, dtype=np.str_)[split_indices])

    return velodyne, labels, poses, cloud_maps, selection_masks


def update_selection_mask(dataset_path: str, sequences: list, split: str) -> list:
    assert split in ['train', 'val']
    selection_masks = []
    for sequence in sequences:
        info_path = os.path.join(dataset_path, 'sequences', f'{sequence:02d}', 'info.h5')
        with h5py.File(info_path, 'r') as f:
            seq_selection_mask = np.asarray(f['selection_mask'])
        selection_masks.append(seq_selection_mask)
    return selection_masks


def crop_sequence_format(data: list, size: int) -> list:
    """Crop the data to the specified size so that sum(N_i for i = 1, ... , S) = size, where
    N_i is the number of samples in the i-th sequence and S is the number of sequences.

    :param data: The data to crop with shape (S, N_i, ...), where
                 S is number of sequences and N number of samples in the i-th sequence
    :param size: The size to crop the data to
    :return: The cropped data
    """

    # Compute the sequence index where the size is located
    seq_idx = 0
    seq_size = len(data[seq_idx])
    while seq_size < size:
        seq_idx += 1
        seq_size += len(data[seq_idx])

    # Compute the sample index where to crop the sequence
    sample_idx = size - seq_size + len(data[seq_idx])

    # Crop the data
    cropped_data = data[:seq_idx]
    cropped_data.append(data[seq_idx][:sample_idx])

    return cropped_data
