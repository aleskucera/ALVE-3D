import os

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from utils.map import map_labels


def load_scan_file(scan_file: str, project_name: str = None) -> dict:
    ret = dict()

    with h5py.File(scan_file, 'r') as f:
        ret['pose'] = np.asarray(f['pose'])
        ret['points'] = np.asarray(f['points'])
        # ret['colors'] = np.asarray(f['colors'])
        ret['remissions'] = np.asarray(f['remissions'])

        ret['labels'] = np.asarray(f['labels']).flatten()
        ret['voxel_map'] = np.asarray(f['voxel_map']).flatten()

    if project_name is not None:
        with h5py.File(scan_file.replace('sequences', project_name), 'r') as f:
            ret['selected_labels'] = np.asarray(f['selected_labels']).flatten()
    return ret


def load_cloud_file(cloud_file: str, project_name: str = None, label_map: dict = None,
                    graph_data: bool = False) -> dict:
    ret = dict()

    with h5py.File(cloud_file, 'r') as f:
        ret['points'] = np.asarray(f['points'])
        # ret['colors'] = np.asarray(f['colors'])
        ret['labels'] = np.asarray(f['labels']).flatten().astype(np.int64)
        ret['objects'] = np.asarray(f['objects']).flatten().astype(np.int64)

        if label_map is not None:
            ret['labels'] = map_labels(ret['labels'], label_map)

        if graph_data:
            ret['edge_sources'] = np.asarray(f['edge_sources']).flatten().astype(np.int64)
            ret['edge_targets'] = np.asarray(f['edge_targets']).flatten().astype(np.int64)
            ret['local_neighbors'] = np.asarray(f['local_neighbors']).flatten().astype(np.int64)

    if project_name is not None:
        with h5py.File(cloud_file.replace('sequences', project_name), 'r') as f:
            ret['selected_edges'] = np.asarray(f['selected_edges']).flatten().astype(bool)
            ret['selected_labels'] = np.asarray(f['selected_labels']).flatten().astype(bool)
            ret['selected_vertices'] = np.asarray(f['selected_vertices']).flatten().astype(bool)
    return ret


def label_voxels_in_cloud(cloud_file: str, project_name: str, voxels: np.ndarray) -> bool:
    with h5py.File(cloud_file.replace('sequences', project_name), 'r+') as f:
        f['selected_labels'][voxels] = 1
        return np.sum(f['selected_labels']) > 0


def label_voxels_in_scan(label_file: str, project_name: str, voxels: np.ndarray) -> bool:
    with h5py.File(label_file, 'r') as f:
        voxel_map = np.asarray(f['voxel_map'])
    with h5py.File(label_file.replace('sequences', project_name), 'r+') as f:
        f['selected_labels'][np.isin(voxel_map, voxels)] = 1
        return np.sum(f['selected_labels']) > 0
