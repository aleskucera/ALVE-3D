import logging

import h5py
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig

from src.datasets import Dataset
from .base_selector import Selector
from .voxel_cloud import VoxelCloud
from src.utils.io import CloudInterface

log = logging.getLogger(__name__)


class VoxelSelector(Selector):
    def __init__(self, cfg: DictConfig, project_name: str, cloud_paths: np.ndarray, device: torch.device):
        super().__init__(cfg, project_name, cloud_paths, device)
        self._initialize()

    def _initialize(self):
        cloud_interface = CloudInterface(self.project_name, self.cfg.ds.learning_map)
        for cloud_id, cloud_path in enumerate(self.cloud_paths):
            with h5py.File(cloud_path, 'r') as f:
                labels = torch.from_numpy(cloud_interface.read_labels(cloud_path))
                surface_variation = torch.from_numpy(cloud_interface.read_surface_variation(cloud_path))
                color_discontinuity = cloud_interface.read_color_discontinuity(cloud_path)
                color_discontinuity = torch.from_numpy(color_discontinuity) if color_discontinuity is not None else None

                num_voxels = f['points'].shape[0]
                self.num_voxels += num_voxels
                self.clouds.append(VoxelCloud(path=cloud_path,
                                              size=num_voxels,
                                              cloud_id=cloud_id,
                                              labels=labels,
                                              diversity_aware=self.diversity_aware,
                                              surface_variation=surface_variation,
                                              color_discontinuity=color_discontinuity))

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5) -> tuple:
        if self.strategy == 'Random':
            return self._select_randomly(percentage)
        else:
            return self._select_by_criterion(dataset, percentage)

    def _select_randomly(self, percentage: float):
        selection_size = self.get_selection_size(percentage)

        labels = torch.tensor([], dtype=torch.long)
        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)

        for cloud in tqdm(self.clouds, desc='Selecting voxels randomly'):
            labels = torch.cat((labels, cloud.labels))
            cloud_map = torch.cat((cloud_map, cloud.ids))
            voxel_map = torch.cat((voxel_map, cloud.voxel_indices))

        return self._choose_voxels(voxel_map, labels, cloud_map, selection_size)

    def _select_by_criterion(self, dataset: Dataset, percentage: float) -> tuple:
        selection_size = self.get_selection_size(percentage)
        self._compute_values(dataset)

        values = torch.tensor([], dtype=torch.float32)
        labels = torch.tensor([], dtype=torch.long)
        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        features = torch.tensor([], dtype=torch.float32)

        for cloud in self.clouds:
            if cloud.values is None:
                continue
            values = torch.cat((values, cloud.values))
            labels = torch.cat((labels, cloud.labels))
            cloud_map = torch.cat((cloud_map, cloud.ids))
            voxel_map = torch.cat((voxel_map, cloud.voxel_indices))
            if cloud.features is not None:
                features = torch.cat((features, cloud.features))

        values = None if values.shape[0] == 0 else values
        features = None if features.shape[0] == 0 else features

        return self._choose_voxels(voxel_map, labels, cloud_map, selection_size, values, features)

    def _choose_voxels(self, voxel_map: torch.Tensor, labels: torch.Tensor, cloud_map: torch.Tensor,
                       selection_size: int, values: torch.Tensor = None, features: torch.Tensor = None) -> tuple:

        normal_order, weighted_order = None, None
        normal_metric_statistics, weighted_metric_statistics = None, None

        if values is None:
            order = torch.randperm(voxel_map.shape[0])
            normal_metric_statistics = None
            weighted_metric_statistics = None
        else:
            normal_order = torch.argsort(values, descending=True)
            weighted_order = self._diversity_aware_order(values, features) if features is not None else None
            order = weighted_order if weighted_order is not None else normal_order

        cloud_map, voxel_map = cloud_map[order], voxel_map[order]
        selected_voxels = voxel_map[:selection_size]
        selected_cloud_map = cloud_map[:selection_size]

        if normal_order is not None:
            normal_values, normal_labels = values[normal_order], labels[normal_order]
            normal_metric_statistics = self._metric_statistics(normal_values, normal_labels, selection_size)
        if weighted_order is not None:
            weighted_values, weighted_labels = values[weighted_order], labels[weighted_order]
            weighted_metric_statistics = self._metric_statistics(weighted_values, weighted_labels, selection_size)

        log.info(f'Choosing voxels from {voxel_map.shape[0]} voxels')
        log.info(f'Selected {selection_size} voxels')
        log.info(f"Order type: {'Weighted' if weighted_order is not None else 'Normal'}")

        voxel_selection = dict()
        for cloud in self.clouds:
            cloud.label_voxels(selected_voxels[selected_cloud_map == cloud.id])
            voxel_selection[cloud.selection_key] = cloud.label_mask

        return voxel_selection, normal_metric_statistics, weighted_metric_statistics
