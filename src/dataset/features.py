import os

import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor

from .utils import open_sequence
from .dataset import SemanticDataset
from src.laserscan import LaserScan
from src import libply_c


def create_features(cfg: DictConfig):
    ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
    laser_scan = LaserScan(label_map=cfg.learning_map, colorize=False, color_map=cfg.color_map_train)

    for i in tqdm(range(len(ds))):
        laser_scan.open_scan(ds.scans[i])
        laser_scan.open_label(ds.labels[i])

        feature_file = ds.scans[i].replace('velodyne', 'features').replace('.bin', '.h5')

        xyz = laser_scan.points
        rgb = laser_scan.color
        labels = laser_scan.sem_label

        xyz, rgb, labels, o = libply_c.prune(xyz.astype('float32'), rgb.astype('uint8'), labels.astype('uint8'),
                                             np.zeros(1, dtype='uint8'), cfg.ds.num_classes, 0)

        graph_nn, local_neighbors = compute_graph_nn(xyz, 20, 5)

        hard_labels = np.argmax(labels, axis=1)
        is_transition = hard_labels[graph_nn['target']] != hard_labels[graph_nn['source']]

        dump, objects = libply_c.connected_comp(xyz.shape[0], graph_nn['source'].astype('uint32'),
                                                graph_nn['target'].astype('uint32'),
                                                (is_transition == 0).astype('uint8'), 0)

        geof = libply_c.compute_geof(xyz, local_neighbors, 5)

        low_points = (xyz[:, 2] - xyz[:, 2].min() < 0.5).nonzero()[0]
        reg = RANSACRegressor(random_state=0).fit(xyz[low_points, :2], xyz[low_points, 2])
        elevation = xyz[:, 2] - reg.predict(xyz[:, :2])

        ma, mi = np.max(xyz[:, :2], axis=0, keepdims=True), np.min(xyz[:, :2], axis=0, keepdims=True)
        xyn = (xyz[:, :2] - mi) / (ma - mi)

        write_structure(feature_file, xyz, rgb, graph_nn, local_neighbors.reshape([xyz.shape[0], 5]),
                        is_transition, labels, objects, geof, elevation, xyn)


def compute_graph_nn(xyz, k_nn1, k_nn2):
    """compute simultaneously 2 knn structures,
    only saves target for knn2
    """
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])

    nn = NearestNeighbors(n_neighbors=k_nn2 + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = np.array(nn.kneighbors(xyz))[..., 1:]

    # ---knn2---
    target2 = (neighbors.flatten()).astype('uint32')

    # ---knn1----
    neighbors = neighbors[:, :k_nn1]
    distances = distances[:, :k_nn1]

    indices = np.arange(0, n_ver)[..., np.newaxis]
    source = np.zeros((n_ver, k_nn1)) + indices

    graph['source'] = source.flatten().astype(np.uint32)
    graph["target"] = neighbors.flatten().astype(np.uint32)
    graph["distances"] = distances.flatten().astype(np.float32)
    return graph, target2


def write_structure(file_name, xyz, rgb, graph_nn, target_local_geometry, is_transition, labels, objects, geof,
                    elevation, xyn):
    """
    save the input point cloud in a format ready for embedding
    """
    # store transition and non-transition edges in two different contiguous memory blocks

    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('xyz', data=xyz, dtype='float32')
    data_file.create_dataset('rgb', data=rgb, dtype='float32')
    data_file.create_dataset('elevation', data=elevation, dtype='float32')
    data_file.create_dataset('xyn', data=xyn, dtype='float32')
    data_file.create_dataset('source', data=graph_nn["source"], dtype='int')
    data_file.create_dataset('target', data=graph_nn["target"], dtype='int')
    data_file.create_dataset('is_transition', data=is_transition, dtype='uint8')
    data_file.create_dataset('target_local_geometry', data=target_local_geometry, dtype='uint32')
    data_file.create_dataset('objects', data=objects, dtype='uint32')
    if len(geof) > 0:
        data_file.create_dataset('geof', data=geof, dtype='float32')
    if len(labels) > 0 and len(labels.shape) > 1 and labels.shape[1] > 1:
        data_file.create_dataset('labels', data=labels, dtype='int32')
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8')


def read_structure(file_name, read_geof):
    """
    read the input point cloud in a format ready for embedding
    """
    data_file = h5py.File(file_name, 'r')
    xyz = np.array(data_file['xyz'], dtype='float32')
    rgb = np.array(data_file['rgb'], dtype='float32')
    elevation = np.array(data_file['elevation'], dtype='float32')
    xyn = np.array(data_file['xyn'], dtype='float32')
    edg_source = np.array(data_file['source'], dtype='int').squeeze()
    edg_target = np.array(data_file['target'], dtype='int').squeeze()
    is_transition = np.array(data_file['is_transition'])
    objects = np.array(data_file['objects'][()])
    labels = np.array(data_file['labels']).squeeze()
    if len(labels.shape) == 0:  # dirty fix
        labels = np.array([0])
    if len(is_transition.shape) == 0:  # dirty fix
        is_transition = np.array([0])
    if read_geof:  # geometry = geometric features
        local_geometry = np.array(data_file['geof'], dtype='float32')
    else:  # geometry = neighborhood structure
        local_geometry = np.array(data_file['target_local_geometry'], dtype='uint32')

    return xyz, rgb, edg_source, edg_target, is_transition, local_geometry, labels, objects, elevation, xyn
