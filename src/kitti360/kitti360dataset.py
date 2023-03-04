import os
import sys
import logging

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset
from numpy.lib.recfunctions import structured_to_unstructured

sys.path.append(os.path.join(os.path.dirname(__file__), '../cut-pursuit/build/src'))

from .ply import read_ply
from src.superpoints.graph import compute_graph_nn_2
from sklearn.linear_model import RANSACRegressor
import libcp
from src.ply_c import libply_c

log = logging.getLogger(__name__)


class KITTI360Dataset(Dataset):
    def __init__(self, cfg: DictConfig, dataset_path: str, split: str, max_points_train: int = 500000,
                 k_nn_adj: int = 5, k_nn_local: int = 20, max_samples: int = 10):
        self.cfg = cfg
        self.split = split
        self.path = dataset_path
        self.max_points_train = max_points_train

        self.k_nn_adj = k_nn_adj
        self.k_nn_local = k_nn_local

        self.max_samples = max_samples

        self.scans = []

        self._init()

    def get_item(self, index):
        scan_path = self.scans[index]
        scan = read_ply(scan_path)

        points = structured_to_unstructured(scan[['x', 'y', 'z']])
        colors = structured_to_unstructured(scan[['red', 'green', 'blue']]) / 255
        x = np.concatenate([points, colors], axis=1)

        y = structured_to_unstructured(scan[['semantic']])

        return x, y

    def __getitem__(self, index):
        scan_path = self.scans[index]
        scan = read_ply(scan_path)

        xyz = structured_to_unstructured(scan[['x', 'y', 'z']])
        rgb = structured_to_unstructured(scan[['red', 'green', 'blue']]) / 255
        labels = structured_to_unstructured(scan[['semantic']])

        # Map labels to train ids
        label_map = np.zeros(max(self.cfg.ds.learning_map.keys()) + 1, dtype=np.uint8)
        for label, value in self.cfg.ds.learning_map.items():
            label_map[label] = value
        labels = label_map[labels]

        # Prune the data
        xyz, rgb, labels, _ = libply_c.prune(xyz.astype('float32'), 0.15, rgb.astype('uint8'),
                                             labels.astype('uint8'),
                                             np.zeros(1, dtype='uint8'), self.cfg.ds.num_classes, 0)

        # Compute the nearest neighbors
        graph_nn, local_neighbors = compute_graph_nn_2(xyz, self.k_nn_adj, self.k_nn_local)
        edg_source, edg_target = graph_nn['source'].astype('int64'), graph_nn['target'].astype('int64')

        _, objects = libply_c.connected_comp(xyz.shape[0], edg_source.astype('uint32'),
                                             edg_target.astype('uint32'),
                                             (labels == 0).astype('uint8'), 0)

        is_transition = objects[edg_source] != objects[edg_target]
        nei = local_neighbors.reshape([xyz.shape[0], self.k_nn_local]).astype('int64')

        selected_ver = np.ones((xyz.shape[0],), dtype=bool)
        if self.split == 'train':
            # Randomly select a subgraph
            selected_edg, selected_ver = libply_c.random_subgraph(xyz.shape[0], edg_source.astype('uint32'),
                                                                  edg_target.astype('uint32'), self.max_points_train)
            # Change the type to bool
            selected_edg = selected_edg.astype(bool)
            selected_ver = selected_ver.astype(bool)

            new_ver_index = -np.ones((xyz.shape[0],), dtype=int)
            new_ver_index[selected_ver.nonzero()] = range(selected_ver.sum())

            edg_source = new_ver_index[edg_source[selected_edg]]
            edg_target = new_ver_index[edg_target[selected_edg]]

            is_transition = is_transition[selected_edg]
            labels = labels[selected_ver,]
            nei = nei[selected_ver,]

        # Compute the elevation
        low_points = (xyz[:, 2] - xyz[:, 2].min() < 0.5).nonzero()[0]
        if low_points.shape[0] > 0:
            reg = RANSACRegressor(random_state=0).fit(xyz[low_points, :2], xyz[low_points, 2])
            elevation = xyz[:, 2] - reg.predict(xyz[:, :2])
        else:
            elevation = np.zeros((xyz.shape[0],), dtype=float)

        # Compute the xyn (normalized x and y)
        ma, mi = np.max(xyz[:, :2], axis=0, keepdims=True), np.min(xyz[:, :2], axis=0, keepdims=True)
        xyn = (xyz[:, :2] - mi) / (ma - mi)
        xyn = xyn[selected_ver,]

        # Compute the local geometry
        clouds = xyz[nei,]
        diameters = np.sqrt(clouds.var(1).sum(1))
        clouds = (clouds - xyz[selected_ver, np.newaxis, :]) / (diameters[:, np.newaxis, np.newaxis] + 1e-10)
        clouds = np.concatenate([clouds, rgb[nei,]], axis=2)
        clouds = clouds.transpose([0, 2, 1])

        # Compute the global geometry
        clouds_global = np.hstack(
            [diameters[:, np.newaxis], elevation[selected_ver, np.newaxis], rgb[selected_ver,], xyn])

        return clouds, clouds_global, labels.astype(np.int32), edg_source, edg_target, is_transition, xyz[selected_ver,]

    def __len__(self):
        return len(self.scans)

    def _init(self):
        split_list = os.path.join(self.path, 'data_3d_semantics', 'train', f'2013_05_28_drive_{self.split}.txt')
        assert os.path.exists(split_list), f'Could not find split list {split_list}'
        with open(split_list, 'r') as f:
            scans = f.read().splitlines()

        self.scans = [os.path.join(self.path, scan) for scan in scans][self.max_samples:]

        log.info(f'Loaded {len(self.scans)} scans from {self.split} split')
