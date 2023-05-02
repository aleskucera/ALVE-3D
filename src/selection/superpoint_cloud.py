import torch

from .base_cloud import Cloud
from src.ply_c import libply_c
from src.utils.io import CloudInterface


class SuperpointCloud(Cloud):
    def __init__(self, path: str, project_name: str, size: int, cloud_id: int, superpoint_map: torch.Tensor,
                 diversity_aware: bool = False):
        super().__init__(path, size, cloud_id, diversity_aware)
        self.project_name = project_name
        self.superpoint_map = superpoint_map

        self.values = None
        self.features = None
        self.cloud_ids = None
        self.superpoint_sizes = None
        self.superpoint_indices = None

    @property
    def num_superpoints(self) -> int:
        return self.superpoint_map.max().item() + 1

    def _average_by_superpoint(self, values: torch.Tensor, features: torch.Tensor = None):
        valid_indices = ~torch.isnan(values)
        values = values[valid_indices]
        average_superpoint_features = None
        superpoint_map = self.superpoint_map[valid_indices]

        superpoints, superpoint_sizes = torch.unique(superpoint_map, return_counts=True)
        average_superpoint_values = torch.full((superpoints.shape[0],), float('nan'), dtype=torch.float32)

        if features is not None:
            features = features[valid_indices]
            average_superpoint_features = torch.full((superpoints.shape[0], self.num_classes), float('nan'),
                                                     dtype=torch.float32)

        # Average the values by superpoint
        for i, superpoint in enumerate(superpoints):
            indices = torch.where(superpoint_map == superpoint)
            superpoint_values = values[indices]
            average_superpoint_values[i] = torch.mean(superpoint_values)
            if features is not None:
                superpoint_features = features[indices]
                average_superpoint_features[i] = torch.mean(superpoint_features, dim=0)

        return superpoints, average_superpoint_values, average_superpoint_features, superpoint_sizes,

    def _save_metric(self, values: torch.Tensor, features: torch.Tensor = None) -> None:
        superpoints, values, features, sizes = self._average_by_superpoint(values, features)
        self.values = values
        self.features = features
        self.superpoint_sizes = sizes
        self.superpoint_indices = superpoints
        self.cloud_ids = torch.full((superpoints.shape[0],), self.id, dtype=torch.long)
        self._reset()

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
