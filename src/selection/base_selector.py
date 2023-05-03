from typing import Any
import logging

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .base_cloud import Cloud
from src.datasets import Dataset

log = logging.getLogger(__name__)


class Selector(object):
    def __init__(self, dataset_path: str, project_name: str, cloud_paths: np.ndarray,
                 device: torch.device, cfg: DictConfig):
        self.device = device
        self.cloud_paths = cloud_paths
        self.dataset_path = dataset_path
        self.project_name = project_name

        self.decay_rate = cfg.active.decay_rate
        self.num_clusters = cfg.active.num_clusters
        self.diversity_aware = cfg.active.diversity_aware
        self.batch_size = cfg.active.batch_size if device.type != 'cpu' else 1

        self.clouds = []
        self.num_voxels = 0
        self.voxels_labeled = 0

    def _initialize(self) -> None:
        raise NotImplementedError

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5) -> dict:
        raise NotImplementedError

    def get_cloud(self, key: Any) -> Cloud:
        for cloud in self.clouds:
            if (isinstance(key, int) or isinstance(key, torch.Tensor)) and cloud.id == key:
                return cloud
            elif isinstance(key, str) and (cloud.path == key or key in cloud.path):
                return cloud

    def get_selection_size(self, percentage: float) -> int:
        return int(self.num_voxels * percentage / 100)

    def load_voxel_selection(self, voxel_selection: dict, dataset: Dataset = None) -> None:
        for cloud_name, label_mask in voxel_selection.items():
            cloud = self.get_cloud(cloud_name)
            voxels = torch.nonzero(label_mask).squeeze(1)
            cloud.label_voxels(voxels, dataset)

    def _calculate_values(self, model: nn.Module, dataset: Dataset, criterion: str, mc_dropout: bool) -> None:
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        model.eval() if not mc_dropout else model.train()
        model.to(self.device)
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Calculating {criterion}'):
                scan_batch, _, voxel_map_batch, cloud_id_batch, end_batch = batch
                scan_batch = scan_batch.to(self.device)

                data = self._get_batch_data(voxel_map_batch, cloud_id_batch, end_batch)
                cloud_ids, split_sizes, voxel_maps, valid_indices, end_indicators = data

                if not mc_dropout:
                    model_outputs = self._get_model_predictions(model, scan_batch, split_sizes, valid_indices)
                else:
                    batch_num_clouds = torch.unique(cloud_id_batch).shape[0]
                    model_outputs = [torch.tensor([], device=self.device) for _ in range(batch_num_clouds)]

                    for j in range(5):
                        model_outputs_it = self._get_model_predictions(model, scan_batch, split_sizes, valid_indices)
                        model_outputs_it = [x.unsqueeze(0) for x in model_outputs_it]
                        for i, model_output in enumerate(model_outputs_it):
                            model_outputs[i] = torch.cat((model_outputs[i], model_output), dim=0)

                for cloud_id, model_output, voxel_map, end in zip(cloud_ids, model_outputs, voxel_maps, end_indicators):
                    cloud = self.get_cloud(cloud_id)
                    cloud.add_predictions(model_output.cpu(), voxel_map, mc_dropout=mc_dropout)
                    if end:
                        log.info('END!!!!!!!!!!!')
                        self._calculate_cloud_values(cloud, criterion)

    def _diversity_aware_order(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        # Sort the values in descending order
        order = torch.argsort(values, descending=True)
        values = values[order]
        features = features[order]

        # Cluster the voxels based on their features
        # kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(features)
        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0, batch_size=5000).fit(features)
        clusters = kmeans.labels_
        clusters = torch.tensor(clusters, dtype=torch.long)

        # Decay the values of the voxels based on their cluster
        unique_clusters, counts = torch.unique(clusters, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            geom_series = [self.decay_rate ** j for j in range(count)]
            values[clusters == cluster] *= torch.tensor(geom_series)

        # Re-sort the voxels based on their weighted values
        weighted_order = order[torch.argsort(values, descending=True)]
        return weighted_order

    @staticmethod
    def _calculate_cloud_values(cloud: Cloud, criterion: str):
        if criterion == 'AverageEntropy':
            cloud.calculate_average_entropies()
        elif criterion == 'ViewpointEntropy':
            cloud.calculate_viewpoint_entropies()
        elif criterion == 'ViewpointVariance':
            cloud.calculate_viewpoint_variances()
        elif criterion == 'EpistemicUncertainty':
            cloud.calculate_epistemic_uncertainties()
        else:
            raise ValueError('Criterion not supported')

    @staticmethod
    def _get_batch_data(voxel_map: torch.Tensor, cloud_map: torch.Tensor, end_batch: torch.Tensor):
        cloud_ids, split_sizes = torch.unique(cloud_map, return_counts=True)
        cloud_voxel_maps = torch.split(voxel_map, list(split_sizes))
        end_indicators = torch.split(end_batch, list(split_sizes))

        voxel_maps = [x.reshape(-1) for x in cloud_voxel_maps]
        valid_indices = [(x != -1) for x in voxel_maps]
        voxel_maps = [x[y] for x, y in zip(voxel_maps, valid_indices)]
        end_indicators = [True if x[-1] else False for x in end_indicators]

        return cloud_ids, split_sizes, voxel_maps, valid_indices, end_indicators

    @staticmethod
    def _get_model_predictions(model: nn.Module, scan_batch: torch.Tensor,
                               split_sizes: torch.Tensor, valid_indices: list):
        model_output = model(scan_batch)
        model_outputs = torch.split(model_output, list(split_sizes))
        model_outputs = [x.permute(0, 2, 3, 1) for x in model_outputs]
        model_outputs = [x.reshape(-1, x.shape[-1]) for x in model_outputs]
        model_outputs = [x[y] for x, y in zip(model_outputs, valid_indices)]
        return model_outputs

    @staticmethod
    def _metric_statistics(values: torch.Tensor, threshold: float) -> dict:
        f = torch.nn.functional.interpolate
        expected_size = min(1000, values.shape[0])
        interp_values = f(values.unsqueeze(0).unsqueeze(0), size=expected_size, mode='linear',
                          align_corners=True).squeeze()
        threshold_index = torch.nonzero(interp_values < threshold, as_tuple=True)[0][0]
        selected_values = interp_values[:threshold_index]
        left_values = interp_values[threshold_index:]
        metric_statistics = {'min': values.min().item(),
                             'max': values.max().item(),
                             'mean': values.mean().item(),
                             'std': values.std().item(),
                             'threshold': threshold,
                             'selected_mean': interp_values[threshold_index:].mean().item(),
                             'selected_std': interp_values[threshold_index:].std().item(),
                             'selected_values': selected_values.tolist(),
                             'left_values': left_values.tolist()}

        return metric_statistics
