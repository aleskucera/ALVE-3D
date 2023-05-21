import logging

import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.io import CloudInterface
from src.datasets import SemanticDataset
from src.utils.cloud import visualize_cloud
from src.utils.map import map_colors
from src.selection import get_selector

log = logging.getLogger(__name__)


def visualize_voxel_selection(cfg: DictConfig) -> None:
    cfg.active.strategy = 'Random'
    cfg.active.cloud_partitions = 'Voxels'
    split = cfg.split if 'split' in cfg else 'train'
    percentage = cfg.percentage if 'percentage' in cfg else 25

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    selector = get_selector(cfg, 'demo', dataset.clouds, device=torch.device('cpu'))

    # Select randomly voxels from the dataset
    selection, _, _ = selector.select(dataset, percentage=percentage)
    selector.load_voxel_selection(selection, dataset)

    # Visualize the selected voxels
    color_map = cfg.ds.color_map_train
    color_map[0] = [175, 175, 175]
    CI = CloudInterface(project_name='demo', label_map=cfg.ds.learning_map)
    for cloud_file in dataset.clouds:
        points = CI.read_points(cloud_file)
        gt_labels = CI.read_labels(cloud_file)
        selection = CI.read_voxel_selection(cloud_file)

        labels = np.zeros_like(gt_labels)
        labels[selection] = gt_labels[selection]
        label_colors = map_colors(labels, color_map)

        visualize_cloud(points, label_colors)


def visualize_superpoint_selection(cfg: DictConfig) -> None:
    cfg.active.strategy = 'Random'
    cfg.active.cloud_partitions = 'Superpoints'
    split = cfg.split if 'split' in cfg else 'train'
    percentage = cfg.percentage if 'percentage' in cfg else 25

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    selector = get_selector(cfg, 'demo', dataset.clouds, device=torch.device('cpu'))

    # Select randomly voxels from the dataset
    selection, _, _ = selector.select(dataset, percentage=percentage)
    selector.load_voxel_selection(selection, dataset)

    # Visualize the selected voxels
    color_map = cfg.ds.color_map_train
    color_map[0] = [175, 175, 175]
    CI = CloudInterface(project_name='demo', label_map=cfg.ds.learning_map)
    for cloud_file in dataset.clouds:
        points = CI.read_points(cloud_file)
        gt_labels = CI.read_labels(cloud_file)
        selection = CI.read_voxel_selection(cloud_file)

        labels = np.zeros_like(gt_labels)
        labels[selection] = gt_labels[selection]
        label_colors = map_colors(labels, color_map)

        visualize_cloud(points, label_colors)


def visualize_scan_selection(cfg: DictConfig) -> None:
    raise NotImplementedError
