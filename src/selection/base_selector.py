from typing import Any
import logging

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from src.models import get_model
from .base_cloud import Cloud
from src.datasets import Dataset
from src.utils.io import CloudInterface

log = logging.getLogger(__name__)


class Selector(object):
    def __init__(self, cfg: DictConfig, project_name: str, cloud_paths: np.ndarray, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.cloud_paths = cloud_paths
        self.project_name = project_name
        self.model = get_model(cfg, device)

        self.strategy = cfg.active.strategy
        self.decay_rate = cfg.active.decay_rate
        self.num_clusters = cfg.active.num_clusters
        self.redal_weights = cfg.active.redal_weights
        self.diversity_aware = cfg.active.diversity_aware
        self.batch_size = cfg.active.batch_size if device.type != 'cpu' else 1
        self.mc_dropout = True if self.strategy == 'EpistemicUncertainty' else False

        self.clouds = []
        self.num_voxels = 0
        self.voxels_labeled = 0

    @property
    def percentage_selected(self) -> float:
        return self.voxels_labeled / self.num_voxels * 100

    def _initialize(self) -> None:
        raise NotImplementedError

    def select(self, dataset: Dataset, percentage: float = 0.5) -> dict:
        raise NotImplementedError

    def get_cloud(self, key: Any) -> Cloud:
        for cloud in self.clouds:
            if (isinstance(key, int) or isinstance(key, torch.Tensor)) and cloud.id == key:
                return cloud
            elif isinstance(key, str) and (cloud.path == key or key in cloud.path):
                return cloud

    def get_selection_size(self, percentage: float) -> int:
        select_percentage = percentage - self.percentage_selected
        log.info(f'Computing selection size for {percentage}% of the dataset.')
        log.info(f'Current percentage of the dataset: {self.percentage_selected}%')
        log.info(f'Percentage of the dataset to be selected: {select_percentage}%')
        return int(self.num_voxels * select_percentage / 100)

    def load_voxel_selection(self, voxel_selection: dict, dataset: Dataset = None) -> None:
        self.voxels_labeled = 0
        for cloud_name, label_mask in voxel_selection.items():
            cloud = self.get_cloud(cloud_name)
            voxels = torch.nonzero(label_mask).squeeze(1)
            self.voxels_labeled += voxels.shape[0]

            cloud.label_mask[voxels] = True
            if dataset is not None:
                dataset.label_voxels(voxels.numpy(), cloud.path)

        log.info(f'Loaded voxel selection with {self.percentage_selected}% of the dataset labeled.')

    def _compute_values(self, dataset: Dataset) -> None:

        dataset.select_mode()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Calculating {self.strategy}'):
                scan_batch, _, voxel_map_batch, cloud_id_batch, end_batch = batch
                scan_batch = scan_batch.to(self.device)

                data = self.__get_batch_data(voxel_map_batch, cloud_id_batch, end_batch)
                cloud_ids, split_sizes, voxel_maps, valid_indices, end_indicators = data

                if not self.mc_dropout:
                    model_outputs = self.__get_model_predictions(scan_batch, split_sizes, valid_indices)
                else:
                    batch_num_clouds = torch.unique(cloud_id_batch).shape[0]
                    model_outputs = [torch.tensor([], device=self.device) for _ in range(batch_num_clouds)]

                    for j in range(5):
                        model_outputs_it = self.__get_model_predictions(scan_batch, split_sizes, valid_indices)
                        model_outputs_it = [x.unsqueeze(0) for x in model_outputs_it]
                        for i, model_output in enumerate(model_outputs_it):
                            model_outputs[i] = torch.cat((model_outputs[i], model_output), dim=0)

                for cloud_id, model_output, voxel_map, end in zip(cloud_ids, model_outputs, voxel_maps, end_indicators):
                    cloud = self.get_cloud(cloud_id)
                    cloud.add_predictions(model_output.cpu(), voxel_map, mc_dropout=self.mc_dropout)
                    if end:
                        self.__compute_cloud_values(cloud)

    def _diversity_aware_order(self, values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        # Sort the values in descending order
        order = torch.argsort(values, descending=True)
        sorted_values = values[order]
        sorted_features = features[order]

        # Cluster the voxels based on their features
        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0, batch_size=10000).fit(sorted_features)
        clusters = kmeans.labels_
        clusters = torch.tensor(clusters, dtype=torch.long)

        # Decay the values of the voxels based on their cluster
        unique_clusters, counts = torch.unique(clusters, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            geom_series = [self.decay_rate ** j for j in range(count)]
            sorted_values[clusters == cluster] *= torch.tensor(geom_series)

        # Re-sort the voxels based on their weighted values
        weighted_order = order[torch.argsort(sorted_values, descending=True)]
        return weighted_order

    def __compute_cloud_values(self, cloud: Cloud):
        if self.strategy == 'ViewpointVariance':
            cloud.compute_viewpoint_variance()
        elif self.strategy == 'EpistemicUncertainty':
            cloud.compute_epistemic_uncertainty()
        elif self.strategy == 'ReDAL':
            cloud.compute_redal_score(self.redal_weights)
        elif self.strategy == 'Entropy':
            cloud.compute_entropy()
        elif self.strategy == 'Margin':
            cloud.compute_margin()
        elif self.strategy == 'Confidence':
            cloud.compute_confidence()
        else:
            raise ValueError('Criterion not supported')

    def __get_model_predictions(self, scan_batch: torch.Tensor,
                                split_sizes: torch.Tensor, valid_indices: list):
        self.model.eval() if not self.mc_dropout else self.model.train()
        model_output = self.model(scan_batch)
        model_outputs = torch.split(model_output, list(split_sizes))
        model_outputs = [x.permute(0, 2, 3, 1) for x in model_outputs]
        model_outputs = [x.reshape(-1, x.shape[-1]) for x in model_outputs]
        model_outputs = [x[y] for x, y in zip(model_outputs, valid_indices)]
        return model_outputs

    @staticmethod
    def __get_batch_data(voxel_map: torch.Tensor, cloud_map: torch.Tensor, end_batch: torch.Tensor):
        cloud_ids, split_sizes = torch.unique(cloud_map, return_counts=True)
        cloud_voxel_maps = torch.split(voxel_map, list(split_sizes))
        end_indicators = torch.split(end_batch, list(split_sizes))

        voxel_maps = [x.reshape(-1) for x in cloud_voxel_maps]
        valid_indices = [(x != -1) for x in voxel_maps]
        voxel_maps = [x[y] for x, y in zip(voxel_maps, valid_indices)]
        end_indicators = [True if x[-1] else False for x in end_indicators]

        return cloud_ids, split_sizes, voxel_maps, valid_indices, end_indicators

    @staticmethod
    def _metric_statistics(values: torch.Tensor, labels: torch.Tensor, split: int) -> dict:

        f = torch.nn.functional.interpolate

        selected_values = values[:split]
        left_values = values[split:]

        sel_interp_size = min(1000, selected_values.shape[0])
        left_interp_size = min(1000, left_values.shape[0])

        # Calculate the count of each class
        selected_labels, label_counts = torch.unique(labels[:split], return_counts=True)
        selected_values = f(selected_values.unsqueeze(0).unsqueeze(0), size=sel_interp_size,
                            mode='linear', align_corners=True).squeeze()
        left_values = f(left_values.unsqueeze(0).unsqueeze(0), size=left_interp_size,
                        mode='linear', align_corners=True).squeeze()
        metric_statistics = {'min': values.min().item(),
                             'max': values.max().item(),
                             'mean': values.mean().item(),
                             'std': values.std().item(),
                             'selected_values': selected_values.tolist(),
                             'left_values': left_values.tolist(),
                             'label_counts': label_counts.tolist(),
                             'selected_labels': selected_labels.tolist()}
        return metric_statistics
