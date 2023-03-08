import os
import sys
import logging

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset
from numpy.lib.recfunctions import structured_to_unstructured

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: Can't import open3d.")

sys.path.append(os.path.join(os.path.dirname(__file__), '../cut-pursuit/build/src'))

from .ply import read_ply
from src.superpoints.graph import compute_graph_nn_2
from src.utils.map import map_labels
from sklearn.linear_model import RANSACRegressor
import libcp
from src.ply_c import libply_c

log = logging.getLogger(__name__)


class KITTI360Dataset(Dataset):
    def __init__(self, cfg: DictConfig, dataset_path: str, split: str, max_points_train: int = 8000,
                 k_nn_adj: int = 5, k_nn_local: int = 20, max_samples: int = 30):
        self.cfg = cfg
        self.split = split
        self.path = dataset_path
        self.max_points_train = max_points_train

        self.k_nn_adj = k_nn_adj
        self.k_nn_local = k_nn_local

        self.max_samples = max_samples

        self.scans = []

        self._init()

    def get_cloud(self, index):
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
        rgb = structured_to_unstructured(scan[['red', 'green', 'blue']])
        labels = structured_to_unstructured(scan[['semantic']])

        # Map labels to train ids
        labels = map_labels(labels, self.cfg.ds.learning_map)

        device = o3d.core.Device("CPU:0")
        pcd = o3d.t.geometry.PointCloud(device)

        pcd.point.positions = o3d.core.Tensor(xyz, o3d.core.float32, device)
        pcd.point.colors = o3d.core.Tensor(rgb, o3d.core.uint8, device)
        pcd.point.labels = o3d.core.Tensor(labels, o3d.core.uint32, device)

        pcd = pcd.voxel_down_sample(voxel_size=0.2)
        xyz, rgb, labels = pcd.point.positions.numpy(), pcd.point.colors.numpy(), pcd.point.labels.numpy()

        # Compute the nearest neighbors
        graph_nn, local_neighbors = compute_graph_nn_2(xyz, self.k_nn_adj, self.k_nn_local)
        edg_source, edg_target = graph_nn['source'].astype('int64'), graph_nn['target'].astype('int64')

        label_transition = labels[edg_source] != labels[edg_target]
        _, objects = libply_c.connected_comp(xyz.shape[0], edg_source.astype('uint32'),
                                             edg_target.astype('uint32'),
                                             (label_transition == 0).astype('uint8'), 0)

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
        low_points = (xyz[:, 2] - xyz[:, 2].min() < 5).nonzero()[0]
        if low_points.shape[0] > 100:
            reg = RANSACRegressor(random_state=0).fit(xyz[low_points, :2], xyz[low_points, 2])
            elevation = xyz[:, 2] - reg.predict(xyz[:, :2])
        else:
            elevation = np.zeros((xyz.shape[0],), dtype=np.float32)

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

        self.scans = [os.path.join(self.path, scan) for scan in scans][:self.max_samples]

        log.info(f'Loaded {len(self.scans)} scans from {self.split} split')


def instances_color_map():
    # make instance colors
    max_inst_id = 10000000
    color_map = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    # force zero to a gray-ish color
    color_map[0] = np.full(3, 0.1)
    return color_map
