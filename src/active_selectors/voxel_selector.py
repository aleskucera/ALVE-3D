import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

from .base_selector import Selector
from .voxel_cloud import VoxelCloud


class VoxelSelector(Selector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device, criterion: str = 'random'):
        super().__init__(dataset_path, cloud_paths, device)
        assert criterion in ['random', 'average_entropy', 'epistemic_uncertainty', 'viewpoint_variance']
        self.criterion = criterion
        self.mc_dropout = True if criterion == 'epistemic_uncertainty' else True

        self._initialize()

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5) -> dict:
        if self.criterion == 'random':
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
        selection_size = self.get_selection_size(dataset, percentage)

        for cloud in self.clouds:
            voxel_mask = dataset.get_voxel_mask(cloud.path, cloud.size)
            voxels = torch.nonzero(torch.from_numpy(voxel_mask)).squeeze(1)

            voxel_map = torch.cat((voxel_map, voxels))
            cloud_map = torch.cat((cloud_map, torch.full((voxels.shape[0],), cloud.id, dtype=torch.long)))

        voxel_selection = self._get_voxel_selection(voxel_map, cloud_map, selection_size)
        return voxel_selection

    def _select_by_criterion(self, dataset: Dataset, model: nn.Module, percentage: float) -> dict:

        values = torch.tensor([], dtype=torch.float32)
        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        selection_size = self.get_selection_size(dataset, percentage)

        self._map_model_predictions(model, dataset, self.mc_dropout)

        for cloud in self.clouds:
            if self.criterion == 'average_entropy':
                cloud_values, cloud_voxel_map, cloud_ids = cloud.get_average_entropies()
            elif self.criterion == 'epistemic_uncertainty':
                cloud_values, cloud_voxel_map, cloud_ids = cloud.get_epistemic_uncertainties()
            elif self.criterion == 'viewpoint_variance':
                cloud_values, cloud_voxel_map, cloud_ids = cloud.get_viewpoint_variances()
            else:
                raise ValueError('Criterion not supported')

            values = torch.cat((values, cloud_values))
            voxel_map = torch.cat((voxel_map, cloud_voxel_map))
            cloud_map = torch.cat((cloud_map, cloud_ids))

            cloud.reset()

        return self._get_voxel_selection(voxel_map, cloud_map, selection_size, values)

    def _get_voxel_selection(self, voxel_map: torch.Tensor, cloud_map: torch.Tensor,
                             selection_size: int, values: torch.Tensor = None) -> dict:
        voxel_selection = dict()

        if values is None:
            order = torch.randperm(voxel_map.shape[0])
        else:
            order = torch.argsort(values, descending=True)
        voxel_map, cloud_map = voxel_map[order], cloud_map[order]
        selected_voxels = voxel_map[:selection_size].cpu()
        selected_cloud_map = cloud_map[:selection_size].cpu()

        for cloud in self.clouds:
            cloud.label_voxels(selected_voxels[selected_cloud_map == cloud.id])
            voxel_selection[cloud.selection_key] = cloud.label_mask
        return voxel_selection
