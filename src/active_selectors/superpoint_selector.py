import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

from .base_selector import Selector
from .superpoint_cloud import SuperpointCloud


class SuperpointSelector(Selector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device, criterion: str = 'random'):
        super().__init__(dataset_path, cloud_paths, device)
        assert criterion in ['random', 'average_entropy', 'epistemic_uncertainty', 'viewpoint_variance']
        self.criterion = criterion
        self.mc_dropout = True if criterion == 'epistemic_uncertainty' else False

        self._initialize()

    def _initialize(self):
        for cloud_id, cloud_path in enumerate(self.cloud_paths):
            with h5py.File(cloud_path, 'r') as f:
                num_voxels = f['points'].shape[0]
                superpoint_map = np.asarray(f['superpoints'], dtype=np.int64)
                superpoint_map = torch.from_numpy(superpoint_map)
                self.num_voxels += num_voxels
                self.clouds.append(SuperpointCloud(cloud_path, num_voxels, cloud_id, superpoint_map))

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5):
        if self.criterion == 'random':
            return self._select_randomly(dataset, percentage)
        else:
            return self._select_by_criterion(dataset, model, percentage)

    def _select_randomly(self, dataset: Dataset, percentage: float):
        selection_size = self.get_selection_size(dataset, percentage)

        cloud_map = torch.tensor([], dtype=torch.long)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)

        for cloud in self.clouds:
            voxel_mask = dataset.get_voxel_mask(cloud.path, cloud.size)
            superpoint_map = cloud.superpoint_map[voxel_mask]
            cloud_superpoint_map, cloud_superpoint_sizes = torch.unique(superpoint_map, return_counts=True)
            cloud_ids = torch.full((cloud_superpoint_map.shape[0],), cloud.id, dtype=torch.long)

            cloud_map = torch.cat((cloud_map, cloud_ids))
            superpoint_map = torch.cat((superpoint_map, cloud_superpoint_map))
            superpoint_sizes = torch.cat((superpoint_sizes, cloud_superpoint_sizes))

        return self._get_voxel_selection(superpoint_map, superpoint_sizes, cloud_map, selection_size)

    def _select_by_criterion(self, dataset: Dataset, model: nn.Module, percentage: float):
        selection_size = self.get_selection_size(dataset, percentage)

        values = torch.tensor([], dtype=torch.float32)
        cloud_map = torch.tensor([], dtype=torch.long)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)

        self._map_model_predictions(model, dataset, self.mc_dropout)

        for cloud in self.clouds:
            if self.criterion == 'average_entropy':
                cloud_values, cloud_superpoint_map, cloud_superpoint_sizes, cloud_ids = cloud.get_average_entropies()
            elif self.criterion == 'epistemic_uncertainty':
                cloud_values, cloud_superpoint_map, cloud_superpoint_sizes, cloud_ids = cloud.get_epistemic_uncertainties()
            elif self.criterion == 'viewpoint_variance':
                cloud_values, cloud_superpoint_map, cloud_superpoint_sizes, cloud_ids = cloud.get_viewpoint_variances()
            else:
                raise ValueError('Criterion not supported')

            values = torch.cat((values, cloud_values))
            cloud_map = torch.cat((cloud_map, cloud_ids))
            superpoint_map = torch.cat((superpoint_map, cloud_superpoint_map))
            superpoint_sizes = torch.cat((superpoint_sizes, cloud_superpoint_sizes))

            cloud.reset()

        return self._get_voxel_selection(superpoint_map, superpoint_sizes, cloud_map, selection_size, values)

    def _get_voxel_selection(self, superpoint_map: torch.Tensor, superpoint_sizes: torch.Tensor,
                             cloud_map: torch.Tensor, selection_size: int, values: torch.Tensor = None):
        if values is None:
            order = torch.randperm(superpoint_map.shape[0])
        else:
            order = torch.argsort(values, descending=True)

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
        return voxel_selection
