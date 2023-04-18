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


class ScanInterface(object):
    def __init__(self, project_name: str = None, label_map: dict = None):
        self.label_map = label_map
        self.project_name = project_name

    @staticmethod
    def read_points(path: str):
        with h5py.File(path, 'r') as f:
            return np.asarray(f['points']).astype(np.float32)

    def read_labels(self, path: str):
        with h5py.File(path, 'r') as f:
            if self.label_map is not None:
                return map_labels(np.asarray(f['labels']).flatten().astype(np.int64), self.label_map)
            return np.asarray(f['labels']).flatten().astype(np.int64)

    @staticmethod
    def read_remissions(path: str):
        with h5py.File(path, 'r') as f:
            return np.asarray(f['remissions']).flatten().astype(np.float32)

    @staticmethod
    def read_pose(path: str):
        with h5py.File(path, 'r') as f:
            return np.asarray(f['pose']).astype(np.float32)

    @staticmethod
    def read_voxel_map(path: str):
        with h5py.File(path, 'r') as f:
            return np.asarray(f['voxel_map']).flatten().astype(np.int64)

    @staticmethod
    def read_colors(path: str):
        with h5py.File(path, 'r') as f:
            if 'colors' in f:
                return np.asarray(f['colors']).astype(np.float32)
            return None

    def read_selected_labels(self, path: str):
        if self.project_name is None:
            return None
        with h5py.File(path.replace('sequences', self.project_name), 'r') as f:
            return np.asarray(f['selected_labels']).flatten().astype(bool)

    def read_scan(self, path: str):
        ret = dict()
        with h5py.File(path, 'r') as f:
            ret['pose'] = np.asarray(f['pose']).astype(np.float32)
            ret['points'] = np.asarray(f['points']).astype(np.float32)
            ret['remissions'] = np.asarray(f['remissions']).astype(np.float32)
            ret['voxel_map'] = np.asarray(f['voxel_map']).flatten().astype(np.int64)

            if 'colors' in f:
                ret['colors'] = np.asarray(f['colors']).astype(np.float32)
            else:
                ret['colors'] = None

            if self.label_map is not None:
                ret['labels'] = map_labels(np.asarray(f['labels']).flatten().astype(np.int64), self.label_map)
            else:
                ret['labels'] = np.asarray(f['labels']).flatten().astype(np.int64)

        if self.project_name is not None:
            with h5py.File(path.replace('sequences', self.project_name), 'r') as f:
                ret['selected_labels'] = np.asarray(f['selected_labels']).flatten().astype(bool)
        return ret

    def select_voxels(self, path: str, voxels: np.ndarray):
        voxel_map = self.read_voxel_map(path)
        with h5py.File(path.replace('sequences', self.project_name), 'r+') as f:
            f['selected_labels'][np.isin(voxel_map, voxels)] = 1
            return np.sum(f['selected_labels']) > 0


class CloudInterface(object):
    def __init__(self, project_name: str = None, label_map: dict = None):
        self.label_map = label_map
        self.project_name = project_name

    @staticmethod
    def read_points(path: str):
        with h5py.File(path, 'r') as f:
            return np.asarray(f['points']).astype(np.float32)

    @staticmethod
    def read_colors(path: str):
        with h5py.File(path, 'r') as f:
            if 'colors' in f:
                return np.asarray(f['colors']).astype(np.float32)
            return None

    @staticmethod
    def read_objects(path: str):
        with h5py.File(path, 'r') as f:
            if 'objects' in f:
                return np.asarray(f['objects']).astype(np.float32)

    def read_labels(self, path: str):
        with h5py.File(path, 'r') as f:
            if self.label_map is not None:
                return map_labels(np.asarray(f['labels']).flatten().astype(np.int64), self.label_map)
            return np.asarray(f['labels']).flatten().astype(np.int64)

    def read_selected_labels(self, path: str):
        if self.project_name is None:
            return None
        with h5py.File(path.replace('sequences', self.project_name), 'r') as f:
            return np.asarray(f['selected_labels']).flatten().astype(bool)

    @staticmethod
    def read_edges(path: str):
        with h5py.File(path, 'r') as f:
            return np.asarray(f['edge_sources']).astype(np.int64), np.asarray(f['edge_targets']).astype(np.int64)

    def read_cloud(self, path: str):
        ret = dict()
        with h5py.File(path, 'r') as f:
            ret['points'] = np.asarray(f['points'])
            ret['labels'] = np.asarray(f['labels']).flatten().astype(np.int64)
            ret['objects'] = np.asarray(f['objects']).flatten().astype(np.int64)

            if 'colors' in f:
                ret['colors'] = np.asarray(f['colors']).astype(np.float32)
            else:
                ret['colors'] = None

            if self.label_map is not None:
                ret['labels'] = map_labels(np.asarray(f['labels']).flatten().astype(np.int64), self.label_map)
            else:
                ret['labels'] = np.asarray(f['labels']).flatten().astype(np.int64)

            ret['edge_sources'] = np.asarray(f['edge_sources']).flatten().astype(np.int64)
            ret['edge_targets'] = np.asarray(f['edge_targets']).flatten().astype(np.int64)
            ret['local_neighbors'] = np.asarray(f['local_neighbors']).flatten().astype(np.int64)

        if self.project_name is not None:
            with h5py.File(path.replace('sequences', self.project_name), 'r') as f:
                ret['selected_edges'] = np.asarray(f['selected_edges']).flatten().astype(bool)
                ret['selected_labels'] = np.asarray(f['selected_labels']).flatten().astype(bool)
                ret['selected_vertices'] = np.asarray(f['selected_vertices']).flatten().astype(bool)
        return ret

    def select_voxels(self, path: str, voxels: np.ndarray):
        with h5py.File(path.replace('sequences', self.project_name), 'r+') as f:
            f['selected_labels'][voxels] = 1

    def select_graph(self, path: str, edges: np.ndarray, vertices: np.ndarray):
        with h5py.File(path.replace('sequences', self.project_name), 'r+') as f:
            f['selected_edges'][edges] = 1
            f['selected_labels'][vertices] = 1
            f['selected_vertices'][vertices] = 1


def load_dataset(dataset_path: str, project_name: str, sequences: list, split: str,
                 al_experiment: bool = False, resume: bool = False) -> dict:
    assert 'project_name' != 'sequences', 'The project name cannot be sequences.'
    ret = {
        'scans': np.array([], dtype=np.str_),
        'cloud_map': np.array([], dtype=np.str_),
        'scan_sequence_map': np.array([], dtype=np.int32),

        'clouds': np.array([], dtype=np.str_),
        'cloud_sequence_map': np.array([], dtype=np.int32)
    }

    sequence_dirs = [os.path.join(dataset_path, 'sequences', f'{s:02d}') for s in sequences]
    for sequence, sequence_dir in tqdm(zip(sequences, sequence_dirs), desc='Loading dataset sequences'):
        info_path = os.path.join(sequence_dir, 'info.h5')
        scans_dir = os.path.join(sequence_dir, 'velodyne')
        clouds_dir = os.path.join(sequence_dir, 'voxel_clouds')

        with h5py.File(info_path, 'r') as f:
            split_samples = np.asarray(f[split]).astype(np.str_)
            seq_clouds = np.asarray(f[f'{split}_clouds']).astype(np.str_)
            seq_cloud_map = __create_cloud_map(seq_clouds)

        seq_scans = np.array([os.path.join(scans_dir, t) for t in split_samples], dtype=np.str_)
        seq_cloud_map = np.array([os.path.join(clouds_dir, t) for t in seq_cloud_map], dtype=np.str_)
        seq_scan_sequence_map = np.full_like(seq_scans, fill_value=sequence, dtype=np.int32)

        seq_clouds = np.array([os.path.join(clouds_dir, t) for t in seq_clouds], dtype=np.str_)
        seq_cloud_sequence_map = np.full_like(seq_clouds, fill_value=sequence, dtype=np.int32)

        ret['scans'] = np.concatenate((ret['scans'], seq_scans)).astype(np.str_)
        ret['cloud_map'] = np.concatenate((ret['cloud_map'], seq_cloud_map)).astype(np.str_)
        ret['scan_sequence_map'] = np.concatenate((ret['scan_sequence_map'], seq_scan_sequence_map))

        ret['clouds'] = np.concatenate((ret['clouds'], seq_clouds)).astype(np.str_)
        ret['cloud_sequence_map'] = np.concatenate((ret['cloud_sequence_map'], seq_cloud_sequence_map))

        for s in sequences:
            os.makedirs(os.path.join(dataset_path, project_name, f'{s:02d}', 'velodyne'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, project_name, f'{s:02d}', 'voxel_clouds'), exist_ok=True)

        if not resume:
            __initialize_dataset(ret['scans'], ret['clouds'], project_name, split, al_experiment)
    return ret


def __initialize_dataset(scans: np.ndarray, clouds: np.ndarray, project_name: str,
                         split: str, al_experiment: bool) -> None:
    for scan in tqdm(scans, desc=f'Initializing sequence scans'):
        with h5py.File(scan, 'r') as f:
            labels = np.asarray(f['labels'])

        with h5py.File(scan.replace('sequences', project_name), 'w') as f:
            if al_experiment and split == 'train':
                f.create_dataset('selected_labels', data=np.zeros_like(labels, dtype=bool))
            else:
                f.create_dataset('selected_labels', data=np.ones_like(labels, dtype=bool))

    for cloud in tqdm(clouds, desc=f'Initializing sequence clouds'):
        with h5py.File(cloud, 'r') as f:
            labels = np.asarray(f['labels'])
            edge_sources = np.asarray(f['edge_sources'])

        with h5py.File(cloud.replace('sequences', project_name), 'w') as f:
            if al_experiment and split == 'train':
                f.create_dataset('selected_labels', data=np.zeros_like(labels, dtype=bool))
                f.create_dataset('selected_vertices', data=np.zeros_like(labels, dtype=bool))
                f.create_dataset('selected_edges', data=np.zeros_like(edge_sources, dtype=bool))
            else:
                f.create_dataset('selected_labels', data=np.ones_like(labels, dtype=bool))
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
