import os
import sys
import time
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../cut-pursuit/build/src'))

import libcp
import numpy as np
from numba import jit, prange
from omegaconf import DictConfig
from sklearn.neighbors import KDTree
from src.datasets import SemanticDataset
from src.utils.io import CloudInterface
from jakteristics import compute_features, FEATURE_NAMES

from src.utils.cloud import nearest_neighbors, nn_graph

log = logging.getLogger(__name__)

CP_CUTOFF = 25
REG_STRENGTH = 10
EDGE_WEIGHT_THRESHOLD = -0.5
FEATURES = ['anisotropy', 'planarity', 'linearity', 'PCA2',
            'sphericity', 'verticality', 'surface_variation']


def create_superpoints(cfg: DictConfig):
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo', cfg=cfg.ds, split='train')
    cloud_interface = CloudInterface()

    for cloud_id, cloud_path in enumerate(dataset.cloud_files):
        points = cloud_interface.read_points(cloud_path)
        colors = cloud_interface.read_colors(cloud_path)
        edge_sources, edge_targets = cloud_interface.read_edges(cloud_path)
        _, superpoint_map = partition_cloud(points, edge_sources, edge_targets, colors)
        cloud_interface.write_superpoints(cloud_path, superpoint_map)

    log.info('Superpoints successfully created')


def partition_cloud(points: np.ndarray, edge_sources: np.ndarray, edge_targets: np.ndarray, colors: np.ndarray = None):
    start = time.time()
    embeddings = compute_features(points.astype(np.double), search_radius=2,
                                  max_k_neighbors=1000, feature_names=FEATURES)
    embeddings[np.isnan(embeddings)] = -1
    if colors is not None:
        embeddings = np.concatenate([embeddings, colors], axis=-1)
    embeddings = np.concatenate([embeddings, points], axis=-1)

    dist = ((embeddings[edge_sources, :] - embeddings[edge_targets, :]) ** 2).sum(1)
    edge_weight = np.exp(dist * EDGE_WEIGHT_THRESHOLD) / np.exp(EDGE_WEIGHT_THRESHOLD)

    components, component_map = libcp.cutpursuit(embeddings.astype(np.float32),
                                                 edge_sources.astype(np.uint32),
                                                 edge_targets.astype(np.uint32),
                                                 edge_weight.astype(np.float32),
                                                 REG_STRENGTH,
                                                 cutoff=CP_CUTOFF, spatial=1,
                                                 weight_decay=0.2)
    log.info(f'Partitioning finished in {time.time() - start:.2f} seconds')
    return components, component_map.astype(np.int64)


def calculate_features(points: np.ndarray) -> dict:
    geometric_features = compute_features(points.astype(np.double), search_radius=2,
                                          max_k_neighbors=1000, feature_names=FEATURE_NAMES)
    geometric_features[np.isnan(geometric_features)] = -1
    ret = {}
    for feature in FEATURE_NAMES:
        ret[feature] = geometric_features[..., FEATURE_NAMES.index(feature)]
    return ret


@jit('float32[:](float32[:,:],float32[:,:],int64[:,:])', nopython=True, cache=True, parallel=True)
def count_colorgrad(rgb, coords, nhoods):
    n = len(nhoods)
    out_arr = np.zeros(n, dtype=np.float32)
    for idx in prange(n):
        cur_rgb = rgb[idx]
        neigh_rgb = rgb[nhoods[idx]]
        diff = np.mean(np.abs(neigh_rgb - cur_rgb))
        out_arr[idx] = diff
    return out_arr


def calculate_color_discontinuity(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    tree = KDTree(points, leaf_size=60)
    nhoods = tree.query(points, k=50, return_distance=False)
    colorgrad = count_colorgrad(colors, points, nhoods)
    colorgrad[colorgrad > 0.1] = 0.1
    return colorgrad


def calculate_surface_variation(points: np.ndarray):
    surface_variation = compute_features(points.astype(np.double), search_radius=2,
                                         max_k_neighbors=1000, feature_names=['surface_variation'])
    surface_variation[np.isnan(surface_variation)] = -1
    return surface_variation[..., 0]
