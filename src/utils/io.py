import os

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from .map import map_labels


def set_paths(cfg: DictConfig, output_dir: str) -> DictConfig:
    cfg.path.output = output_dir
    for i in cfg.path:
        cfg.path[i] = to_absolute_path(cfg.path[i])
    cfg.ds.path = to_absolute_path(cfg.ds.path)
    return cfg


def load_scan_file(scan_file: str) -> dict:
    ret = dict()

    with h5py.File(scan_file, 'r') as f:
        ret['points'] = np.asarray(f['points'])
        ret['colors'] = np.asarray(f['colors'])
        ret['remissions'] = np.asarray(f['remissions'])
    return ret


def load_label_file(label_file: str, project_name: str, label_map: dict = None) -> dict:
    ret = dict()

    with h5py.File(label_file, 'r') as f:
        ret['labels'] = np.asarray(f['labels']).flatten()
        ret['voxel_map'] = np.asarray(f['voxel_map']).flatten()

        if label_map is not None:
            ret['labels'] = map_labels(ret['labels'], label_map)

    with h5py.File(label_file.replace('sequences', project_name), 'r+') as f:
        ret['label_mask'] = np.asarray(f['label_mask']).flatten()

    return ret


def load_cloud_file(cloud_file: str, project_name: str, label_map: dict = None, graph_data: bool = False) -> dict:
    ret = dict()

    with h5py.File(cloud_file, 'r') as f:
        ret['points'] = np.asarray(f['points'])
        ret['colors'] = np.asarray(f['colors'])
        ret['labels'] = np.asarray(f['labels']).flatten().astype(np.int64)
        ret['objects'] = np.asarray(f['objects']).flatten().astype(np.int64)
        ret['superpoints'] = np.asarray(f['superpoints']).flatten().astype(np.int64)

        if label_map is not None:
            ret['labels'] = map_labels(ret['labels'], label_map)

        if graph_data:
            ret['edge_sources'] = np.asarray(f['edge_sources']).flatten().astype(np.int64)
            ret['edge_targets'] = np.asarray(f['edge_targets']).flatten().astype(np.int64)
            ret['edge_transitions'] = np.asarray(f['edge_transitions']).flatten().astype(bool)
            ret['local_neighbors'] = np.asarray(f['local_neighbors']).flatten().astype(np.int64)

    with h5py.File(cloud_file.replace('sequences', project_name), 'r') as f:
        ret['label_mask'] = np.asarray(f['label_mask']).flatten().astype(bool)
        ret['selected_edges'] = np.asarray(f['selected_edges']).flatten().astype(bool)
        ret['selected_vertices'] = np.asarray(f['selected_vertices']).flatten().astype(bool)

    return ret


def label_voxels_in_cloud(cloud_file: str, project_name: str, voxels: np.ndarray) -> bool:
    with h5py.File(cloud_file.replace('sequences', project_name), 'r+') as f:
        f['label_mask'][voxels] = 1
        return np.sum(f['label_mask']) > 0


def label_voxels_in_scan(label_file: str, project_name: str, voxels: np.ndarray) -> bool:
    with h5py.File(label_file, 'r') as f:
        voxel_map = np.asarray(f['voxel_map'])
    with h5py.File(label_file.replace('sequences', project_name), 'r+') as f:
        f['label_mask'][np.isin(voxel_map, voxels)] = 1
        return np.sum(f['label_mask']) > 0


def load_dataset(dataset_path: str, project_name: str, sequences: list, split: str,
                 al_experiment: bool = False, resume: bool = False) -> dict:
    assert 'project_name' != 'sequences', 'The project name cannot be sequences.'
    for s in sequences:
        os.makedirs(os.path.join(dataset_path, project_name, f'{s:02d}', 'velodyne'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, project_name, f'{s:02d}', 'voxel_clouds'), exist_ok=True)

    ret = {
        'scans': np.array([], dtype=np.str_),
        'labels': np.array([], dtype=np.str_),
        'cloud_map': np.array([], dtype=np.str_),
        'scan_sequence_map': np.array([], dtype=np.int32),

        'clouds': np.array([], dtype=np.str_),
        'cloud_sequence_map': np.array([], dtype=np.int32)
    }

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]
    for sequence, sequence_dir in tqdm(zip(sequences, sequence_dirs), desc='Loading dataset sequences'):
        info_path = os.path.join(sequence_dir, 'info.h5')
        labels_dir = os.path.join(sequence_dir, 'velodyne')
        scans_dir = os.path.join(sequence_dir, 'velodyne')
        clouds_dir = os.path.join(sequence_dir, 'voxel_clouds')

        with h5py.File(info_path, 'r') as f:
            split_samples = np.asarray(f[split]).astype(np.str_)
            seq_clouds = np.asarray(f[f'{split}_clouds']).astype(np.str_)
            seq_cloud_map = __create_cloud_map(seq_clouds)

        seq_scans = np.array([os.path.join(scans_dir, t) for t in split_samples], dtype=np.str_)
        seq_labels = np.array([os.path.join(labels_dir, t) for t in split_samples], dtype=np.str_)
        seq_cloud_map = np.array([os.path.join(clouds_dir, t) for t in seq_cloud_map], dtype=np.str_)
        seq_scan_sequence_map = np.full_like(seq_scans, fill_value=sequence, dtype=np.int32)

        seq_clouds = np.array([os.path.join(clouds_dir, t) for t in seq_clouds], dtype=np.str_)
        seq_cloud_sequence_map = np.full_like(seq_clouds, fill_value=sequence, dtype=np.int32)

        ret['scans'] = np.concatenate((ret['scans'], seq_scans)).astype(np.str_)
        ret['labels'] = np.concatenate((ret['labels'], seq_labels)).astype(np.str_)
        ret['cloud_map'] = np.concatenate((ret['cloud_map'], seq_cloud_map)).astype(np.str_)
        ret['scan_sequence_map'] = np.concatenate((ret['scan_sequence_map'], seq_scan_sequence_map))

        ret['clouds'] = np.concatenate((ret['clouds'], seq_clouds)).astype(np.str_)
        ret['cloud_sequence_map'] = np.concatenate((ret['cloud_sequence_map'], seq_cloud_sequence_map))

        if not resume:
            __initialize_dataset(ret['labels'], ret['clouds'], project_name, split, al_experiment)
    return ret


def __initialize_dataset(labels: np.ndarray, clouds: np.ndarray, project_name: str,
                         split: str, al_experiment: bool) -> None:
    print(project_name)
    for label in tqdm(labels, desc=f'Initializing sequence labels'):
        with h5py.File(label, 'r') as f:
            labels = np.asarray(f['labels'], dtype=bool)

        with h5py.File(label.replace('sequences', project_name), 'w') as f:
            if al_experiment and split == 'train':
                f.create_dataset('label_mask', data=np.zeros_like(labels, dtype=bool))
            else:
                f.create_dataset('label_mask', data=np.ones_like(labels, dtype=bool))

    for cloud in tqdm(clouds, desc=f'Initializing sequence clouds'):
        with h5py.File(cloud, 'r') as f:
            labels = np.asarray(f['labels'])
            edge_sources = np.asarray(f['edge_sources'])

        with h5py.File(cloud.replace('sequences', project_name), 'w') as f:
            if al_experiment and split == 'train':
                f.create_dataset('label_mask', data=np.zeros_like(labels, dtype=bool))
                f.create_dataset('selected_vertices', data=np.zeros_like(labels, dtype=bool))
                f.create_dataset('selected_edges', data=np.zeros_like(edge_sources, dtype=bool))
            else:
                f.create_dataset('label_mask', data=np.ones_like(labels, dtype=bool))
                f.create_dataset('selected_vertices', data=np.ones_like(labels, dtype=bool))
                f.create_dataset('selected_edges', data=np.ones_like(edge_sources, dtype=bool))


def __create_cloud_map(clouds: np.ndarray) -> np.ndarray:
    cloud_map = np.array([], dtype=np.str_)
    for cloud_file in sorted(clouds):
        cloud_name = os.path.splitext(cloud_file)[0]
        split_cloud_name = cloud_name.split('_')
        cloud_size = int(split_cloud_name[1]) - int(split_cloud_name[0]) + 1
        cloud_map = np.concatenate((cloud_map, np.tile(cloud_file, cloud_size)), axis=0).astype(np.str_)
    return cloud_map
