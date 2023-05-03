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
            superpoint_map = torch.from_numpy(cloud_interface.read_superpoints(cloud_path))
            surface_variation = torch.from_numpy(cloud_interface.read_surface_variation(cloud_path))
            color_discontinuity = cloud_interface.read_color_discontinuity(cloud_path)
            color_discontinuity = torch.from_numpy(color_discontinuity) if color_discontinuity is not None else None
            num_voxels = superpoint_map.shape[0]
            self.num_voxels += num_voxels
            self.clouds.append(SuperpointCloud(path=cloud_path, size=num_voxels,
                                               cloud_id=cloud_id, superpoint_map=superpoint_map,
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
            cloud_map = torch.cat((cloud_map, cloud.cloud_ids))
            superpoint_map = torch.cat((superpoint_map, cloud.superpoint_map))
            superpoint_sizes = torch.cat((superpoint_sizes, cloud.superpoint_sizes))
            if cloud.features is not None:
                features = torch.cat((features, cloud.features))

        values = None if values.shape[0] == 0 else values
        features = None if features.shape[0] == 0 else features

        return self._choose_voxels(superpoint_map, superpoint_sizes, cloud_map, selection_size, values, features)

    def _choose_voxels(self, superpoint_map: torch.Tensor, superpoint_sizes: torch.Tensor,
                       cloud_map: torch.Tensor, selection_size: int, values: torch.Tensor = None,
                       features: torch.Tensor = None) -> tuple:
        metric_statistics = None
        if values is None:
            order = torch.randperm(superpoint_map.shape[0])
        elif features is None:
            order = torch.argsort(values, descending=True)
        else:
            order = self._diversity_aware_order(values, features)

        cloud_map = cloud_map[order]
        superpoint_map = superpoint_map[order]
        superpoint_sizes = superpoint_sizes[order]
        superpoint_sizes = torch.cumsum(superpoint_sizes, dim=0)

        log.info(f"Choosing superpoints from {superpoint_map.shape[0]} superpoints")
        log.info(f"Superpoint sizes: {superpoint_sizes}")
        log.info(f"Selection size: {selection_size}")

        selected_superpoints = superpoint_map[superpoint_sizes < selection_size]
        selected_cloud_map = cloud_map[superpoint_sizes < selection_size]

        log.info(f"Selected superpoints: {selected_superpoints.shape[0]}")

        # if values is not None:
        #     values = values[order]
        #     threshold = values[superpoint_sizes < selection_size][-1]
        #     metric_statistics = self._metric_statistics(values, threshold)

        voxel_selection = dict()
        for cloud in self.clouds:
            superpoints = selected_superpoints[selected_cloud_map == cloud.id]
            for superpoint in superpoints:
                voxels = torch.nonzero(cloud.superpoint_map == superpoint).squeeze(1)
                cloud.label_voxels(voxels)
            voxel_selection[cloud.selection_key] = cloud.label_mask
        return voxel_selection, metric_statistics
