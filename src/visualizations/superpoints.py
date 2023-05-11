import logging

import numpy as np
from omegaconf import DictConfig

from src.utils.io import CloudInterface
from src.datasets import SemanticDataset
from src.utils.cloud import visualize_cloud
from src.process import partition_cloud, calculate_features
from src.utils.map import colorize_values, colorize_instances

log = logging.getLogger(__name__)


def visualize_feature(cfg: DictConfig) -> None:
    split = cfg.split if 'split' in cfg else 'train'
    feature = cfg.feature if 'feature' in cfg else 'planarity'

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    cloud_interface = CloudInterface()

    for cloud_file in dataset.clouds:
        points = cloud_interface.read_points(cloud_file)

        # Compute features
        feature = calculate_features(points)[feature]

        # Visualize feature
        rng = (np.min(feature[feature != -1]), np.max(feature))
        feature_colors = colorize_values(feature, color_map='viridis', data_range=rng, ignore=(-1,))
        visualize_cloud(points, feature_colors)


def visualize_superpoints(cfg: DictConfig) -> None:
    split = cfg.split if 'split' in cfg else 'train'

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    cloud_interface = CloudInterface()

    for cloud_file in dataset.clouds:
        points = cloud_interface.read_points(cloud_file)
        colors = cloud_interface.read_colors(cloud_file)
        edge_sources, edge_targets = cloud_interface.read_edges(cloud_file)

        components, component_map = partition_cloud(points=points, colors=colors,
                                                    edge_sources=edge_sources, edge_targets=edge_targets)

        superpoint_colors = colorize_instances(component_map)
        visualize_cloud(points, superpoint_colors)
