import torch

from .base_cloud import Cloud
from src.ply_c import libply_c
from src.utils.io import CloudInterface


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

        superpoints, superpoint_sizes = torch.unique(self.superpoint_map, return_counts=True)
        average_superpoint_values = torch.full((superpoints.shape[0],), float('nan'), dtype=torch.float32)

        # Average the values by superpoint
        for superpoint in superpoints:
            indices = torch.where(self.superpoint_map == superpoint)
            superpoint_values = values[indices]
            valid_superpoint_values = superpoint_values[~torch.isnan(superpoint_values)]
            if len(valid_superpoint_values) > 0:
                average_superpoint_values[superpoint] = torch.mean(valid_superpoint_values)
        valid_indices = ~torch.isnan(average_superpoint_values)
        return superpoints[valid_indices], average_superpoint_values[valid_indices], superpoint_sizes[valid_indices]

    def _save_values(self, values: torch.Tensor):
        superpoints, superpoint_values, superpoint_sizes = self._average_by_superpoint(values)
        self.values = superpoint_values
        self.superpoint_indices = superpoints
        self.superpoint_sizes = superpoint_sizes
        self.cloud_ids = torch.full((superpoints.shape[0],), self.id, dtype=torch.long)

    def subgraph(self, size: int):
        cloud_interface = CloudInterface(self.project_name)
        points = cloud_interface.read_points(self.path)
        edge_sources, edge_targets = cloud_interface.read_edges(self.path)
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
