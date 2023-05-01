import torch

from .base_cloud import Cloud


class VoxelCloud(Cloud):
    def __init__(self, path: str, size: int, cloud_id: int, diversity_aware: bool = False):
        super().__init__(path, size, cloud_id, diversity_aware)

        self.values = None
        self.features = None
        self.cloud_ids = None
        self.voxel_indices = None

    def _save_values(self, values: torch.Tensor, features: torch.Tensor = None):
        valid_indices = ~torch.isnan(values)

        self.values = values[valid_indices]
        self.features = features[valid_indices] if features is not None else None
        self.voxel_indices = torch.arange(self.size, dtype=torch.long)[valid_indices]
        self.cloud_ids = torch.full((self.size,), self.id, dtype=torch.long)[valid_indices]

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
