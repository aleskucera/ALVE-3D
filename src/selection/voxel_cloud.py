import torch

from .base_cloud import Cloud


class VoxelCloud(Cloud):
    def __init__(self, path: str, size: int, cloud_id: int,
                 diversity_aware: bool,
                 surface_variation: torch.Tensor,
                 color_discontinuity: torch.Tensor = None):
        super().__init__(path, size, cloud_id, diversity_aware,
                         surface_variation, color_discontinuity)
        self.values = None
        self.features = None
        self.voxel_indices = torch.arange(self.size, dtype=torch.long)
        self.cloud_ids = torch.full((self.size,), self.id, dtype=torch.long)

    def _save_values(self, values: torch.Tensor, features: torch.Tensor = None):
        self.values = values
        self.features = features

    def __str__(self):
        ret = f'\nVoxelCloud:\n' \
              f'\t - Cloud ID = {self.id}, \n' \
              f'\t - Cloud path = {self.path}, \n' \
              f'\t - Number of voxels in cloud = {self.size}\n' \
              f'\t - Number of model predictions = {self.predictions.shape[0]}\n'
        if self.num_classes > 0:
            ret += f'\t - Number of semantic classes = {self.num_classes}\n'
        ret += f'\t - Percentage labeled = {torch.sum(self.label_mask) / self.size * 100:.2f}%\n'
        return ret
