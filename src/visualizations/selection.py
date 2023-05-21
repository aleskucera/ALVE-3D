import logging

import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.io import CloudInterface
from src.datasets import SemanticDataset
from src.utils.map import map_colors
from src.selection import get_selector
from src.utils.wb import pull_artifact
from src.utils.cloud import visualize_cloud, visualize_cloud_values

log = logging.getLogger(__name__)


def visualize_voxel_selection(cfg: DictConfig) -> None:
    cfg.active.batch_size = 2
    cfg.active.cloud_partitions = 'Voxels'
    split = cfg.split if 'split' in cfg else 'train'
    percentage = cfg.percentage if 'percentage' in cfg else 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    selector = get_selector(cfg, 'demo', dataset.clouds, device=device)

    # Load the model
    model_state_dict = pull_artifact(cfg.active.model_artifact, device=device)
    selector.model.load_state_dict(model_state_dict)

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
    cfg.active.batch_size = 2
    cfg.active.cloud_partitions = 'Superpoints'
    split = cfg.split if 'split' in cfg else 'train'
    percentage = cfg.percentage if 'percentage' in cfg else 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    selector = get_selector(cfg, 'demo', dataset.clouds, device=device)

    # Load the model
    model_state_dict = pull_artifact(cfg.active.model_artifact, device=device)
    selector.model.load_state_dict(model_state_dict)

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


def visualize_uncertainty_score_voxels(cfg: DictConfig) -> None:
    cfg.active.batch_size = 2
    cfg.active.cloud_partitions = 'Voxels'
    split = cfg.split if 'split' in cfg else 'train'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    selector = get_selector(cfg, 'demo', dataset.clouds, device=device)

    # Load the model
    model_state_dict = pull_artifact(cfg.active.model_artifact, device=device)
    selector.model.load_state_dict(model_state_dict)

    # Compute the uncertainty score for each voxel
    selector._compute_values(dataset)

    CI = CloudInterface(label_map=cfg.ds.learning_map)
    for cloud in selector.clouds:
        values = cloud.values
        points = CI.read_points(cloud.path)
        visualize_cloud_values(points, values.numpy())


def visualize_uncertainty_score_superpoints(cfg: DictConfig) -> None:
    cfg.active.batch_size = 2
    cfg.active.cloud_partitions = 'Superpoints'
    split = cfg.split if 'split' in cfg else 'train'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    selector = get_selector(cfg, 'demo', dataset.clouds, device=device)

    # Load the model
    model_state_dict = pull_artifact(cfg.active.model_artifact, device=device)
    selector.model.load_state_dict(model_state_dict)

    # Compute the uncertainty score for each voxel
    selector._compute_values(dataset)
    CI = CloudInterface(label_map=cfg.ds.learning_map)
    for cloud in selector.clouds:
        superpoint_values = cloud.values
        superpoint_map = cloud.superpoint_map
        mapped_values = torch.zeros_like(superpoint_map, dtype=torch.float32)
        for superpoint in torch.unique(superpoint_map):
            mapped_values[superpoint_map == superpoint] = superpoint_values[superpoint]
        points = CI.read_points(cloud.path)
        visualize_cloud_values(points, mapped_values.numpy())
