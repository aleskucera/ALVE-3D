import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

from .base_selector import Selector
from .superpoint_cloud import SuperpointCloud


class SuperpointSelector(Selector):
    def __init__(self, dataset_path: str, project_name: str, cloud_paths: np.ndarray,
                 device: torch.device, criterion: str, batch_size: int):
        super().__init__(dataset_path, project_name, cloud_paths, device, batch_size)
        self.criterion = criterion
        self.mc_dropout = True if criterion == 'EpistemicUncertainty' else False
        self._initialize()

    def _initialize(self):
        for cloud_id, cloud_path in enumerate(self.cloud_paths):
            with h5py.File(cloud_path, 'r') as f:
                num_voxels = f['points'].shape[0]
                superpoint_map = np.asarray(f['superpoints'], dtype=np.int64)
                superpoint_map = torch.from_numpy(superpoint_map)
                self.num_voxels += num_voxels
                self.clouds.append(SuperpointCloud(cloud_path, self.project_name, num_voxels, cloud_id, superpoint_map))

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5) -> tuple:
        if self.criterion == 'Random':
            return self._select_randomly(dataset, percentage)
        else:
            return self._select_by_criterion(dataset, model, percentage)

    def _select_randomly(self, dataset: Dataset, percentage: float) -> tuple:
        selection_size = self.get_selection_size(dataset, percentage)

        cloud_map = torch.tensor([], dtype=torch.long)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)

        for cloud in self.clouds:
            voxel_mask = dataset.get_voxel_mask(cloud.path, cloud.size)
            cloud_superpoint_map = cloud.superpoint_map[voxel_mask]
            superpoints, cloud_superpoint_sizes = torch.unique(cloud_superpoint_map, return_counts=True)
            cloud_ids = torch.full((superpoints.shape[0],), cloud.id, dtype=torch.long)

            cloud_map = torch.cat((cloud_map, cloud_ids))
            superpoint_map = torch.cat((superpoint_map, superpoints))
            superpoint_sizes = torch.cat((superpoint_sizes, cloud_superpoint_sizes))

        return self._choose_voxels(superpoint_map, superpoint_sizes, cloud_map, selection_size)

    def _select_random_graphs(self, dataset: Dataset, percentage: float) -> tuple:
        selection_size = self.get_selection_size(percentage)
        print(selection_size)
        subgraph_size = max(10000, selection_size // 10)
        sizes = [subgraph_size] * (selection_size // subgraph_size)
        sizes.append(selection_size % subgraph_size)
        print(sizes)

        raise NotImplementedError()

    def _select_by_criterion(self, dataset: Dataset, model: nn.Module, percentage: float) -> tuple:
        values = torch.tensor([], dtype=torch.float32)
        cloud_map = torch.tensor([], dtype=torch.long)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)
        selection_size = self.get_selection_size(dataset, percentage)

        self._calculate_values(model, dataset, self.criterion, self.mc_dropout)

        for cloud in self.clouds:
            values = torch.cat((values, cloud.values))
            cloud_map = torch.cat((cloud_map, cloud.cloud_ids))
            superpoint_map = torch.cat((superpoint_map, cloud.superpoint_indices))
            superpoint_sizes = torch.cat((superpoint_sizes, cloud.superpoint_sizes))

        return self._choose_voxels(superpoint_map, superpoint_sizes, cloud_map, selection_size, values)

    def _choose_voxels(self, superpoint_map: torch.Tensor, superpoint_sizes: torch.Tensor,
                       cloud_map: torch.Tensor, selection_size: int, values: torch.Tensor = None) -> tuple:

        if values is None:
            order = torch.randperm(superpoint_map.shape[0])
            metric_statistics = None
        else:
            order = torch.argsort(values, descending=True)
            values = values[order]
            threshold = values[selection_size - 1]
            metric_statistics = self._metric_statistics(values, threshold)

        cloud_map = cloud_map[order]
        superpoint_map = superpoint_map[order]
        superpoint_sizes = superpoint_sizes[order]

        superpoint_sizes = torch.cumsum(superpoint_sizes, dim=0)

        selected_superpoints = superpoint_map[superpoint_sizes < selection_size]
        selected_cloud_map = cloud_map[superpoint_sizes < selection_size]

        voxel_selection = dict()
        for cloud in self.clouds:
            superpoints = selected_superpoints[selected_cloud_map == cloud.id]
            for superpoint in superpoints:
                voxels = torch.nonzero(cloud.superpoint_map == superpoint).squeeze(1)
                cloud.label_voxels(voxels)
            voxel_selection[cloud.selection_key] = cloud.label_mask
        return voxel_selection, metric_statistics
