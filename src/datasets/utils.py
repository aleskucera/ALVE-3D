import os
import logging

import h5py
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def initialize_dataset(dataset_path: str, project_name: str, sequences: list, split: str, active: bool) -> None:
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

    log.info(f'Initializing dataset: {dataset_path} with mode {"active" if active else "normal"}')

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]
    for sequence_dir in sequence_dirs:
        info_path = os.path.join(sequence_dir, 'info.h5')
        labels_dir = os.path.join(sequence_dir, 'labels')
        clouds_dir = os.path.join(sequence_dir, 'voxel_clouds')

        with h5py.File(info_path, 'r') as f:
            if split == 'train':
                samples = np.asarray(f['train']).astype(np.str_)
                clouds = np.asarray(f['train_clouds']).astype(np.str_)
            elif split == 'val':
                samples = np.asarray(f['val']).astype(np.str_)
                clouds = np.asarray(f['val_clouds']).astype(np.str_)

            labels = np.array([os.path.join(labels_dir, t) for t in samples], dtype=np.str_)
            clouds = np.array([os.path.join(clouds_dir, t) for t in clouds], dtype=np.str_)

        if split == 'train':
            with h5py.File(info_path.replace('sequences', project_name), 'w') as f:
                if active:
                    f.create_dataset('selection_mask', data=np.zeros_like(samples, dtype=bool))
                else:
                    f.create_dataset('selection_mask', data=np.ones_like(samples, dtype=bool))

        for label in tqdm(labels, desc=f'Initializing labels for sequence {sequence_dir}'):
            with h5py.File(label, 'r') as f:
                label_mask = np.asarray(f['label_mask'], dtype=bool)
            with h5py.File(label.replace('sequences', project_name), 'w') as f:
                if active and split == 'train':
                    f.create_dataset('label_mask', data=np.zeros_like(label_mask, dtype=bool))
                else:
                    f.create_dataset('label_mask', data=np.ones_like(label_mask, dtype=bool))

        for cloud in tqdm(clouds, desc=f'Initializing clouds for sequence {sequence_dir}'):
            with h5py.File(cloud, 'r') as f:
                label_mask = np.asarray(f['label_mask'], dtype=bool)
            with h5py.File(cloud.replace('sequences', project_name), 'w') as f:
                if active and split == 'train':
                    f.create_dataset('label_mask', data=np.zeros_like(label_mask, dtype=bool))
                else:
                    f.create_dataset('label_mask', data=np.ones_like(label_mask, dtype=bool))


def load_semantic_dataset(dataset_path: str, project_name: str, sequences: list, split: str,
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
    assert 'project_name' != 'sequences', 'The project name cannot be sequences.'
    for s in sequences:
        os.makedirs(os.path.join(dataset_path, project_name, f'{s:02d}'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, project_name, f'{s:02d}', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, project_name, f'{s:02d}', 'voxel_clouds'), exist_ok=True)

    if not resume:
        initialize_dataset(dataset_path, project_name, sequences, split, active)

    log.info(f'Loading dataset: {dataset_path} split: {split} with mode {"active" if active else "normal"}')

    scans = np.array([], dtype=np.str_)
    labels = np.array([], dtype=np.str_)
    cloud_map = np.array([], dtype=np.str_)
    sequence_map = np.array([], dtype=np.int32)
    selection_mask = np.array([], dtype=bool)

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]
    for sequence, sequence_dir in tqdm(zip(sequences, sequence_dirs), desc='Loading dataset sequences'):
        info_path = os.path.join(sequence_dir, 'info.h5')
        labels_dir = os.path.join(sequence_dir, 'labels')
        scans_dir = os.path.join(sequence_dir, 'velodyne')
        clouds_dir = os.path.join(sequence_dir, 'voxel_clouds')

        with h5py.File(info_path, 'r') as f:
            split_samples = np.asarray(f[split]).astype(np.str_)
            seq_clouds = np.asarray(f[f'{split}_clouds']).astype(np.str_)
            seq_cloud_map = create_cloud_map(seq_clouds)
            if split == 'train':
                seq_selection_mask = np.asarray(f['selection_mask']).astype(bool)
            else:
                seq_selection_mask = np.ones_like(split_samples, dtype=bool)

        seq_scans = np.array([os.path.join(scans_dir, t) for t in split_samples], dtype=np.str_)
        seq_labels = np.array([os.path.join(labels_dir, t) for t in split_samples], dtype=np.str_)
        seq_cloud_map = np.array([os.path.join(clouds_dir, t) for t in seq_cloud_map], dtype=np.str_)

        scans = np.concatenate((scans, seq_scans), axis=0).astype(np.str_)
        labels = np.concatenate((labels, seq_labels), axis=0).astype(np.str_)
        cloud_map = np.concatenate((cloud_map, seq_cloud_map), axis=0).astype(np.str_)
        selection_mask = np.concatenate((selection_mask, seq_selection_mask), axis=0).astype(bool)
        sequence_map = np.concatenate((sequence_map, np.full_like(split_samples, fill_value=sequence)), axis=0).astype(
            np.int32)

    return scans, labels, sequence_map, cloud_map, selection_mask


def create_cloud_map(clouds: np.ndarray) -> np.ndarray:
    cloud_map = np.array([], dtype=np.str_)
    for cloud_file in sorted(clouds):
        cloud_name = os.path.splitext(cloud_file)[0]
        split_cloud_name = cloud_name.split('_')
        cloud_size = int(split_cloud_name[1]) - int(split_cloud_name[0]) + 1
        cloud_map = np.concatenate((cloud_map, np.tile(cloud_file, cloud_size)), axis=0).astype(np.str_)
    return cloud_map
