import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import RANSACRegressor

from .base_dataset import Dataset


class PartitionDataset(Dataset):
    def __init__(self,
                 split: str,
                 cfg: DictConfig,
                 dataset_path: str,
                 project_name: str,
                 resume: bool = False,
                 num_clouds: int = None,
                 sequences: iter = None,
                 al_experiment: bool = False,
                 selection_mode: bool = False):
        super().__init__(split, cfg, dataset_path,
                         project_name, resume, num_clouds,
                         sequences, al_experiment, selection_mode)
        self.parser_type = 'partition'

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
        cloud_data = self.CI.read_cloud(self.clouds[idx])
        edge_sources, edge_targets = cloud_data['edge_sources'], cloud_data['edge_targets']
        points, colors, objects = cloud_data['points'], cloud_data['colors'], cloud_data['objects']
        edge_transitions, local_neighbors = cloud_data['edge_transitions'], cloud_data['local_neighbors']
        selected_edges, selected_vertices = cloud_data['selected_edges'], cloud_data['selected_vertices']
        local_neighbors = local_neighbors.reshape((points.shape[0], 20))

        new_ver_index = -np.ones((points.shape[0],), dtype=int)
        new_ver_index[selected_vertices.nonzero()] = range(selected_vertices.sum())

        edge_sources = new_ver_index[edge_sources[selected_edges]]
        edge_targets = new_ver_index[edge_targets[selected_edges]]

        edge_transitions = edge_transitions[selected_edges]
        local_neighbors = local_neighbors[selected_vertices]

        low_points = (points[:, 2] - points[:, 2].min() < 5).nonzero()[0]
        reg = RANSACRegressor(random_state=0).fit(points[low_points, :2], points[low_points, 2])
        elevation = points[:, 2] - reg.predict(points[:, :2])

        # Compute the xyn (normalized x and y)
        ma, mi = np.max(points[:, :2], axis=0, keepdims=True), np.min(points[:, :2], axis=0, keepdims=True)
        xyn = (points[:, :2] - mi) / (ma - mi)
        xyn = xyn[selected_vertices]

        # Compute the local geometry
        clouds = points[local_neighbors]
        diameters = np.sqrt(clouds.var(1).sum(1))
        clouds = (clouds - points[selected_vertices, np.newaxis, :]) / (diameters[:, np.newaxis, np.newaxis] + 1e-10)
        clouds = np.concatenate([clouds, points[local_neighbors]], axis=2)
        clouds = clouds.transpose([0, 2, 1])

        # Compute the global geometry
        clouds_global = np.hstack(
            [diameters[:, np.newaxis], elevation[selected_vertices, np.newaxis], points[selected_vertices,], xyn])

        return clouds, clouds_global, objects[selected_vertices], edge_sources, edge_targets, edge_transitions

    def __len__(self):
        return len(self.clouds)

    def __str__(self):
        experiment = 'Active Learning' if self.al_experiment else 'Full Training'
        ret = f'\n{self.__class__.__name__} ({self.split}):' \
              f'\n\t- Dataset size: {self.__len__()}' \
              f'\n\t- Project name: {self.project_name}' \
              f'\n\t- Usage: {experiment}\n'
        return ret
