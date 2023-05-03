import logging
import torch
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .base_selector import Selector
from src.utils.io import CloudInterface
from .superpoint_cloud import SuperpointCloud

log = logging.getLogger(__name__)


class SuperpointSelector(Selector):
    def __init__(self, dataset_path: str, project_name: str, cloud_paths: np.ndarray,
                 device: torch.device, criterion: str, cfg: DictConfig):
        super().__init__(dataset_path, project_name, cloud_paths, device, cfg)
        self.criterion = criterion
        self.mc_dropout = True if criterion == 'EpistemicUncertainty' else False
        self._initialize()

    def _initialize(self):
        cloud_interface = CloudInterface(self.project_name)
        for cloud_id, cloud_path in enumerate(self.cloud_paths):
            labels = cloud_interface.read_labels(cloud_path)
            superpoint_map = torch.from_numpy(cloud_interface.read_superpoints(cloud_path))
            surface_variation = torch.from_numpy(cloud_interface.read_surface_variation(cloud_path))
            color_discontinuity = cloud_interface.read_color_discontinuity(cloud_path)
            color_discontinuity = torch.from_numpy(color_discontinuity) if color_discontinuity is not None else None
            num_voxels = superpoint_map.shape[0]
            self.num_voxels += num_voxels
            self.clouds.append(SuperpointCloud(path=cloud_path,
                                               size=num_voxels,
                                               cloud_id=cloud_id,
                                               superpoint_map=superpoint_map,
                                               labels=labels,
                                               diversity_aware=self.diversity_aware,
                                               surface_variation=surface_variation,
                                               color_discontinuity=color_discontinuity))

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5) -> tuple:
        if self.criterion == 'Random':
            return self._select_randomly(percentage)
        else:
            return self._select_by_criterion(dataset, model, percentage)

    def _select_randomly(self, percentage: float) -> tuple:
        selection_size = self.get_selection_size(percentage)

        cloud_map = torch.tensor([], dtype=torch.long)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)

        for cloud in self.clouds:
            superpoints, cloud_superpoint_sizes = torch.unique(cloud.superpoint_map, return_counts=True)
            cloud_ids = torch.full((superpoints.shape[0],), cloud.id, dtype=torch.long)

            cloud_map = torch.cat((cloud_map, cloud_ids))
            superpoint_map = torch.cat((superpoint_map, superpoints))
            superpoint_sizes = torch.cat((superpoint_sizes, cloud_superpoint_sizes))

        return self._choose_voxels(superpoint_map, superpoint_sizes, cloud_map, selection_size)

    def _select_by_criterion(self, dataset: Dataset, model: nn.Module, percentage: float) -> tuple:
        values = torch.tensor([], dtype=torch.float32)
        labels = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        features = torch.tensor([], dtype=torch.float32)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)
        selection_size = self.get_selection_size(percentage)

        self._compute_values(model, dataset, self.criterion, self.mc_dropout)

        for cloud in self.clouds:
            if cloud.values is None:
                continue
            values = torch.cat((values, cloud.values))
            labels = torch.cat((labels, cloud.superpoint_labels))
            cloud_map = torch.cat((cloud_map, cloud.cloud_ids))
            superpoint_map = torch.cat((superpoint_map, cloud.superpoint_indices))
            superpoint_sizes = torch.cat((superpoint_sizes, cloud.superpoint_sizes))
            if cloud.features is not None:
                features = torch.cat((features, cloud.features))

        values = None if values.shape[0] == 0 else values
        features = None if features.shape[0] == 0 else features

        return self._choose_voxels(superpoint_map, superpoint_sizes, labels, cloud_map, selection_size, values,
                                   features)

    def _choose_voxels(self, superpoint_map: torch.Tensor, superpoint_sizes: torch.Tensor, labels: torch.Tensor,
                       cloud_map: torch.Tensor, selection_size: int, values: torch.Tensor = None,
                       features: torch.Tensor = None) -> tuple:

        normal_order, weighted_order = None, None
        normal_metric_statistics, weighted_metric_statistics = None, None

        if values is None:
            order = torch.randperm(superpoint_map.shape[0])
            normal_metric_statistics = None
            weighted_metric_statistics = None
        else:
            normal_order = torch.argsort(values, descending=True)
            weighted_order = self._diversity_aware_order(values, features) if features is not None else None
            order = weighted_order if weighted_order is not None else normal_order

        cloud_map, superpoint_map = cloud_map[order], superpoint_map[order]
        superpoint_cum_sizes = torch.cumsum(superpoint_sizes[order], dim=0)
        selected_superpoints = superpoint_map[superpoint_cum_sizes < selection_size]
        selected_cloud_map = cloud_map[superpoint_cum_sizes < selection_size]

        if normal_order is not None:
            normal_values, normal_labels = values[normal_order], labels[normal_order]
            normal_cum_sizes = torch.cumsum(superpoint_sizes[normal_order], dim=0)
            normal_split = torch.nonzero(normal_cum_sizes >= selection_size, as_tuple=False)[0, 0]
            normal_metric_statistics = self._metric_statistics(normal_values, normal_labels, normal_split)
            log.info(f"Normal metric statistics: {normal_metric_statistics}")
        if weighted_order is not None:
            weighted_values, weighted_labels = values[weighted_order], labels[weighted_order]
            weighted_cum_sizes = torch.cumsum(superpoint_sizes[weighted_order], dim=0)
            weighted_split = torch.nonzero(weighted_cum_sizes >= selection_size, as_tuple=False)[0, 0]
            weighted_metric_statistics = self._metric_statistics(weighted_values, weighted_labels, weighted_split)
            log.info(f"Weighted metric statistics: {weighted_metric_statistics}")

        log.info(f"Choosing superpoints from {superpoint_map.shape[0]} superpoints")
        log.info(f"Selected {selected_superpoints.shape[0]} superpoints")

        voxel_selection = dict()
        for cloud in self.clouds:
            superpoints = selected_superpoints[selected_cloud_map == cloud.id]
            for superpoint in superpoints:
                voxels = torch.nonzero(cloud.superpoint_map == superpoint).squeeze(1)
                cloud.label_voxels(voxels)
            voxel_selection[cloud.selection_key] = cloud.label_mask
        return voxel_selection, normal_metric_statistics, weighted_metric_statistics
