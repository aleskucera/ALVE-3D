import torch
from .base_cloud import Cloud

from src.ply_c import libply_c
from src.utils import load_cloud_file


class SuperpointCloud(Cloud):
    def __init__(self, path: str, project_name: str, size: int, cloud_id: int, superpoint_map: torch.Tensor):
        super().__init__(path, size, cloud_id)
        self.project_name = project_name
        self.superpoint_map = superpoint_map

        self.values = None
        self.cloud_ids = None
        self.superpoint_sizes = None
        self.superpoint_indices = None

    @property
    def num_superpoints(self) -> int:
        return self.superpoint_map.max().item() + 1

    def _average_by_superpoint(self, values: torch.Tensor):
        superpoint_sizes = torch.zeros((self.num_superpoints,), dtype=torch.long)
        average_superpoint_values = torch.full((self.num_superpoints,), float('nan'), dtype=torch.float32)

        superpoints, sizes = torch.unique(self.superpoint_map, return_counts=True)
        superpoint_sizes[superpoints] = sizes

        # Average the values by superpoint
        for superpoint in torch.unique(self.superpoint_map):
            indices = torch.where(self.superpoint_map == superpoint)
            superpoint_values = values[indices]
            valid_superpoint_values = superpoint_values[~torch.isnan(superpoint_values)]
            if len(valid_superpoint_values) > 0:
                average_superpoint_values[superpoint] = torch.mean(valid_superpoint_values)
        return average_superpoint_values, superpoint_sizes

    def _save_values(self, values: torch.Tensor):
        superpoint_values, superpoint_sizes = self._average_by_superpoint(values)
        valid_indices = ~torch.isnan(superpoint_values)

        self.values = superpoint_values[valid_indices]
        self.superpoint_sizes = superpoint_sizes[valid_indices]
        self.superpoint_indices = self.superpoint_map[valid_indices]
        self.cloud_ids = torch.full((self.num_superpoints,), self.id, dtype=torch.long)[valid_indices]

    def subgraph(self, size: int):
        cloud_data = load_cloud_file(self.path, self.project_name)
        points = cloud_data['points']
        edge_sources = cloud_data['edge_sources']
        edge_targets = cloud_data['edge_targets']
        selected_edg, selected_ver = libply_c.random_subgraph(points.shape[0], edge_sources.astype('uint32'),
                                                              edge_targets.astype('uint32'), size)

        return selected_edg.astype(bool), selected_ver.astype(bool)

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
