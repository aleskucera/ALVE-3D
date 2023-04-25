import h5py
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from src.datasets import Dataset
from .base_selector import Selector
from .voxel_cloud import VoxelCloud


class VoxelSelector(Selector):
    def __init__(self, dataset_path: str, project_name: str, cloud_paths: np.ndarray,
                 device: torch.device, criterion: str, batch_size: int):
        super().__init__(dataset_path, project_name, cloud_paths, device, batch_size)
        self.criterion = criterion
        self.mc_dropout = True if criterion == 'EpistemicUncertainty' else False
        self._initialize()

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5) -> tuple:
        if self.criterion == 'Random':
            return self._select_randomly(dataset, percentage)
        else:
            return self._select_by_criterion(dataset, model, percentage)

    def _initialize(self):
        for cloud_id, cloud_path in enumerate(self.cloud_paths):
            with h5py.File(cloud_path, 'r') as f:
                num_voxels = f['points'].shape[0]
                self.num_voxels += num_voxels
                self.clouds.append(VoxelCloud(cloud_path, num_voxels, cloud_id))

    def _select_randomly(self, dataset: Dataset, percentage: float):

        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        selection_size = self.get_selection_size(percentage)

        for cloud in tqdm(self.clouds, desc='Selecting voxels randomly'):
            voxel_map = torch.cat((voxel_map, torch.arange(cloud.size)))
            cloud_map = torch.cat((cloud_map, torch.full((cloud.size,), cloud.id, dtype=torch.long)))

        return self._choose_voxels(voxel_map, cloud_map, selection_size)

    def _select_by_criterion(self, dataset: Dataset, model: nn.Module, percentage: float) -> tuple:

        values = torch.tensor([], dtype=torch.float32)
        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        selection_size = self.get_selection_size(percentage)

        self._calculate_values(model, dataset, self.criterion, self.mc_dropout)

        for cloud in self.clouds:
            values = torch.cat((values, cloud.values))
            voxel_map = torch.cat((voxel_map, cloud.voxel_indices))
            cloud_map = torch.cat((cloud_map, cloud.cloud_ids))

        return self._choose_voxels(voxel_map, cloud_map, selection_size, values)

    def _choose_voxels(self, voxel_map: torch.Tensor, cloud_map: torch.Tensor,
                       selection_size: int, values: torch.Tensor = None) -> tuple:
        voxel_selection = dict()

        if values is None:
            order = torch.randperm(voxel_map.shape[0])
            metric_statistics = None
        else:
            order = torch.argsort(values, descending=True)
            values = values[order]
            threshold = values[selection_size]
            metric_statistics = self._metric_statistics(values, threshold)

        voxel_map, cloud_map = voxel_map[order], cloud_map[order]
        selected_voxels = voxel_map[:selection_size]
        selected_cloud_map = cloud_map[:selection_size]

        for cloud in self.clouds:
            cloud.label_voxels(selected_voxels[selected_cloud_map == cloud.id])
            voxel_selection[cloud.selection_key] = cloud.label_mask

        return voxel_selection, metric_statistics
