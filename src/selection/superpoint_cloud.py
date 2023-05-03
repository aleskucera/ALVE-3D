import torch
from torch_scatter import scatter_mean

from .base_cloud import Cloud
from src.ply_c import libply_c
from src.utils.io import CloudInterface


class SuperpointCloud(Cloud):
    def __init__(self, path: str, size: int, cloud_id: int,
                 superpoint_map: torch.Tensor, diversity_aware: bool,
                 surface_variation: torch.Tensor,
                 color_discontinuity: torch.Tensor = None):
        super().__init__(path, size, cloud_id, diversity_aware,
                         surface_variation, color_discontinuity)
        self.superpoint_map = superpoint_map

        self.values = None
        self.features = None
        self.superpoint_sizes = None
        self.cloud_ids = torch.full((self.num_superpoints,), self.id, dtype=torch.long)

    @property
    def num_superpoints(self) -> int:
        return self.superpoint_map.max().item() + 1

    def _save_metric(self, values: torch.Tensor, features: torch.Tensor = None) -> None:
        self.superpoint_sizes = torch.bincount(self.superpoint_map)
        self.values = scatter_mean(values, self.superpoint_map, dim=0)
        if features is not None:
            self.superpoint_features = scatter_mean(features, self.superpoint_map, dim=0)

    # def _average_by_superpoint(self, values: torch.Tensor, features: torch.Tensor = None):
    #     superpoint_sizes = torch.bincount(self.superpoint_map)
    #     superpoint_values = scatter_mean(values, self.superpoint_map, dim=0)
    #     if features is not None:
    #         superpoint_features = scatter_mean(features, self.superpoint_map, dim=0)
    #         return superpoint_values, superpoint_features, superpoint_sizes
    #     return superpoint_values, superpoint_sizes

    def __str__(self):
        ret = f'\nSuperpointCloud:\n' \
              f'\t - Cloud ID = {self.id}, \n' \
              f'\t - Cloud path = {self.path}, \n' \
              f'\t - Number of voxels in cloud = {self.size}\n' \
              f'\t - Number of superpoints in cloud = {self.num_superpoints}\n' \
              f'\t - Number of model predictions = {self.predictions.shape[0]}\n'
        if self.num_classes > 0:
            ret += f'\t - Number of semantic classes = {self.num_classes}\n'
        ret += f'\t - Percentage labeled = {torch.sum(self.label_mask) / self.size * 100:.2f}%\n'
        return ret
