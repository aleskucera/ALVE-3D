import os

import h5py
import numpy as np
from tqdm import tqdm


def initialize_dataset(dataset_path: str, sequences: list, active: bool = False) -> None:
    """ Initialize semantic samples for a given split and mode. The initialization is done by setting
    the selection mask and the label mask to 0 or 1 depending on the mode.

    If active is True, the train selection mask and the label mask are set to 0. This means that
    the samples are not selected and the labels are not available.

    If active is False, the train selection mask and the label mask are set to 1. This means that
    the samples are selected and the labels are available. (default)

    Validation samples are always selected and the labels are available.

    :param dataset_path: Path to the dataset.
    :param sequences: List of sequences to initialize.
    :param active: If True, the samples are initialized as not selected and the labels are not available.
    """

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]
    for sequence_dir in sequence_dirs:
        info_path = os.path.join(sequence_dir, 'info.h5')
        labels_dir = os.path.join(sequence_dir, 'labels')
        clouds_dir = os.path.join(sequence_dir, 'voxel_clouds')

        # Update the selection mask in the info file and load the train samples
        with h5py.File(info_path, 'r+') as f:
            train_samples = np.asarray(f['train']).astype(np.str_)
            train_clouds = np.asarray(f['train_clouds']).astype(np.str_)
            train_labels = np.array([os.path.join(labels_dir, t) for t in train_samples], dtype=np.str_)
            train_clouds = np.array([os.path.join(clouds_dir, t) for t in train_clouds], dtype=np.str_)
            selection_mask = np.zeros_like(f['selection_mask']) if active else np.ones_like(f['selection_mask'])
            f['selection_mask'][...] = selection_mask

        # Open the label files and set the label mask to 0 or 1
        for label_path in tqdm(train_labels):
            with h5py.File(label_path, 'r+') as f:
                f['label_mask'][...] = np.zeros_like(f['label_mask']) if active else np.ones_like(f['label_mask'])

        # Open the cloud files and set the label mask to 0 or 1
        for cloud in train_clouds:
            with h5py.File(cloud, 'r+') as f:
                f['label_mask'][...] = np.zeros_like(f['label_mask']) if active else np.ones_like(f['label_mask'])


def load_semantic_dataset(dataset_path: str, sequences: list, split: str,
                          active: bool = False, resume: bool = False) -> tuple:
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
    :param resume: If True, the dataset is not initialized and the dataset is loaded as it is stored on disk.
    :return: Tuple containing the scans, labels, poses, sequence map, cloud map, and selection mask.
    """

    if not resume and split == 'train':
        initialize_dataset(dataset_path, sequences, active)

    scans = np.array([], dtype=np.str_)
    labels = np.array([], dtype=np.str_)
    cloud_map = np.array([], dtype=np.int32)
    sequence_map = np.array([], dtype=np.int32)
    selection_mask = np.array([], dtype=bool)

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]
    for sequence, sequence_dir in zip(sequences, sequence_dirs):
        info_path = os.path.join(sequence_dir, 'info.h5')
        labels_dir = os.path.join(sequence_dir, 'labels')
        scans_dir = os.path.join(sequence_dir, 'velodyne')

        with h5py.File(info_path, 'r+') as f:
            split_samples = np.asarray(f[split]).astype(np.str_)
            seq_cloud_map = create_cloud_map(np.asarray(f[f'{split}_clouds']).astype(np.str_))
            seq_selection_mask = np.asarray(f['selection_mask']).astype(bool) if split == 'train' else np.ones_like(
                split_samples)

        seq_scans = np.array([os.path.join(scans_dir, t) for t in split_samples], dtype=np.str_)
        seq_labels = np.array([os.path.join(labels_dir, t) for t in split_samples], dtype=np.str_)

        scans = np.concatenate((scans, seq_scans), axis=0)
        labels = np.concatenate((labels, seq_labels), axis=0)
        cloud_map = np.concatenate((cloud_map, seq_cloud_map), axis=0)
        selection_mask = np.concatenate((selection_mask, seq_selection_mask), axis=0)
        sequence_map = np.concatenate((sequence_map, np.full_like(split_samples, fill_value=sequence)), axis=0)

        print(f'Scans type: {scans.dtype}')
        print(f'Labels type: {labels.dtype}')
        print(f'Cloud map type: {cloud_map.dtype}')
        print(f'Sequence map type: {sequence_map.dtype}')
        print(f'Selection mask type: {selection_mask.dtype}')

    return scans, labels, sequence_map, cloud_map, selection_mask


def create_cloud_map(clouds: np.ndarray) -> np.ndarray:
    """ Create a cloud map which maps every sample to the specific cloud it belongs to. The index of the
    cloud is relative to the sequence. The cloud map is created by executing a following steps:

    1. For each cloud name in the clouds array, the cloud range is extracted. The cloud range is the
       first and last sample index of the cloud.
    2. For every cloud create a full array of the cloud index with the same length as the cloud range.
    3. Concatenate all the cloud index arrays and return the result.

    Example cloud file: 000023_002452.h5

    :param clouds: Array of cloud names.
    :return: Array of cloud indices.
    """

    cloud_map = np.array([], dtype=np.int32)
    for cloud_id, cloud_file in enumerate(sorted(clouds)):
        cloud_name = os.path.splitext(cloud_file)[0]
        split_cloud_name = cloud_name.split('_')
        cloud_size = int(split_cloud_name[1]) - int(split_cloud_name[0]) + 1
        cloud_map = np.concatenate((cloud_map, np.full(cloud_size, fill_value=cloud_id, dtype=np.int32)), axis=0)

    return cloud_map

# def load_sample_selection_mask(dataset_path: str, sequences: list, split: str) -> np.ndarray:
#     """ Load the sample selection mask for a given split.
#
#     :param dataset_path: Path to the dataset.
#     :param sequences: List of sequences to load.
#     :param split: Split to load.
#     :return: Array of selection mask.
#     """
#
#     assert split in ['train', 'val']
#
#     selection_mask = np.array([], dtype=bool)
#
#     sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]
#
#     for sequence_dir in sequence_dirs:
#         info_path = os.path.join(sequence_dir, 'info.h5')
#         with h5py.File(info_path, 'r') as f:
#             if split == 'train':
#                 seq_selection_mask = np.asarray(f['selection_mask'])
#                 selection_mask = np.concatenate((selection_mask, seq_selection_mask), axis=0)
#             elif split == 'val':
#                 selection_mask = np.concatenate((selection_mask, np.ones_like(f[split])), axis=0)
#     return selection_mask
