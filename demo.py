#!/usr/bin/env python
import os
import logging

import torch
import wandb
import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.datasets import SemanticDataset
from src.utils.io import set_paths, ScanInterface
from src.kitti360 import KITTI360Converter
from src.utils.map import colorize_values
from src.utils.visualize import plot
from src.utils.filter import filter_scan

from src.visualizations.dataset import visualize_scans, visualize_clouds, visualize_statistics
from src.visualizations.superpoints import visualize_feature, visualize_superpoints
from src.visualizations.experiment import visualize_model_comparison, visualize_learning, visualize_loss_comparison, \
    visualize_baseline

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    # ==================== DATASET VISUALIZATIONS ====================

    if cfg.option == 'dataset_scans':
        visualize_scans(cfg)
    elif cfg.option == 'dataset_clouds':
        visualize_clouds(cfg)
    elif cfg.option == 'dataset_statistics':
        visualize_statistics(cfg)
    elif cfg.option == 'augmentation':
        raise NotImplementedError

    # ==================== SUPERPOINT VISUALIZATIONS ====================

    elif cfg.option == 'feature':
        visualize_feature(cfg)
    elif cfg.option == 'superpoints':
        visualize_superpoints(cfg)

    # ==================== EXPERIMENT VISUALIZATIONS ====================

    elif cfg.option == 'model_comparison':
        visualize_model_comparison(cfg)
    elif cfg.option == 'loss_comparison':
        visualize_loss_comparison(cfg)
    elif cfg.option == 'baseline':
        visualize_baseline(cfg)
    elif cfg.option == 'strategy_comparison':
        visualize_learning(cfg)

    # ==================== FILTERING ====================

    elif cfg.option == 'filters':
        visualize_filters(cfg)

    # ==================== MODEL PREDICTIONS ====================

    elif cfg.option == 'model_predictions':
        raise NotImplementedError

    # ==================== DATASET CONVERSION ====================

    elif cfg.option == 'kitti360_conversion':
        converter = KITTI360Converter(cfg)
        converter.visualize()
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def visualize_filters(cfg: DictConfig):
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)

    scan_interface = ScanInterface()
    scan = dataset.scans[491]
    points = scan_interface.read_points(scan)
    radius = points[:, 2]
    colors = colorize_values(radius, color_map='inferno', data_range=(np.min(radius), np.max(radius)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

    dist = filter_scan(points, 'distance')
    rad = filter_scan(points, 'radius')
    stat = filter_scan(points, 'statistical')

    new_colors = np.full(colors.shape, [0.7, 0.7, 0.7])
    new_colors[dist] = np.array([131, 56, 236]) / 255

    # Filter from points indices rad and dist
    valid_points = np.setdiff1d(np.arange(points.shape[0]), rad)
    valid_points = np.setdiff1d(valid_points, dist)

    pcd.points = o3d.utility.Vector3dVector(points[valid_points])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid_points])
    o3d.visualization.draw_geometries([pcd])

    # # Remove points that are too far away
    # _, f1_mask = filter_distant_points(points, 30)
    # f1_colors = colors
    # f1_colors[~f1_mask] = [1, 0, 0]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(f1_colors)
    # o3d.visualization.draw_geometries([pcd])
    #
    # # Remove radius outliers
    # _, f2_indices = filter_radius_outliers(points, 10, 0.5)
    # f2_colors = np.full_like(colors, [1, 0, 0])
    # f2_colors[f2_indices] = colors[f2_indices]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(f2_colors)
    # o3d.visualization.draw_geometries([pcd])
    #
    # # Remove statistical outliers
    # _, f3_indices = filter_statistical_outliers(points, 10, 1.8)
    # f3_colors = np.full_like(colors, [1, 0, 0])
    # f3_colors[f3_indices] = colors[f3_indices]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(f3_colors)
    # o3d.visualization.draw_geometries([pcd])


def visualize_model_training(cfg: DictConfig) -> None:
    ignore_index = cfg.ds.ignore_index
    label_names = [v for k, v in cfg.ds.labels_train.items() if k != ignore_index]
    history_artifact = 'aleskucera/Semantic Segmentation Models Comparison/History_SalsaNext_SemanticKITTI:v0'
    history_file = history_artifact.split('/')[-1].split(':')[0]

    api = wandb.Api()
    history = api.artifact(history_artifact)
    history_path = os.path.join(history.download(), f'{history_file}.pt')
    history = torch.load(history_path)
    plot({'Loss Train': history['loss_train'], 'Loss Val': history['loss_val']}, 'Epoch', 'Value', 'Loss')
    plot({'Accuracy Train': history['accuracy_train'], 'Accuracy Val': history['accuracy_val']},
         'Epoch', 'Value', 'Accuracy')
    plot({'MIoU Train': history['miou_train'], 'IoU Val': history['miou_val']}, 'Epoch', 'Value', 'MIoU')

    class_iou_dict = {}
    class_ious = np.stack([t.numpy() for t in history['class_iou']])
    class_ious = np.delete(class_ious, ignore_index, axis=-1)
    for i, label_name in enumerate(label_names):
        class_iou_dict[label_name] = class_ious[:, i]
    plot(class_iou_dict, 'Epoch', 'Value', 'Class IoU')

    # TODO: Add confusion matrix of the best epoch
    # TODO: Add gradient flow of the last epoch


def visualize_model_experiment(cfg: DictConfig) -> None:
    ignore_index = cfg.ds.ignore_index
    label_names = [v for k, v in cfg.ds.labels_train.items() if k != ignore_index]
    histories = {'SalsaNext': 'aleskucera/Semantic Segmentation Models Comparison/History_SalsaNext_SemanticKITTI:v0',
                 'DeepLabV3+': 'aleskucera/Semantic Segmentation Models Comparison/History_DeepLabV3_SemanticKITTI:v0'}

    files = {model_name: history_artifact.split('/')[-1].split(':')[0] for model_name, history_artifact in
             histories.items()}

    api = wandb.Api()
    histories = {model_name: torch.load(os.path.join(api.artifact(history_artifact).download(), f'{file}.pt'))
                 for (model_name, history_artifact), file in zip(histories.items(), files.values())}
    losses = dict()
    for model_name, history in histories.items():
        losses[f'{model_name} Loss'] = {'Train': history['loss_train'], 'Val': history['loss_val']}

    mious = dict()
    for model_name, history in histories.items():
        mious[f'{model_name} MIoU'] = {'Train': history['miou_train'], 'Val': history['miou_val']}

    accuracies = dict()
    for model_name, history in histories.items():
        accuracies[f'{model_name} Accuracy'] = {'Train': history['accuracy_train'], 'Val': history['accuracy_val']}

    class_ious = dict()
    for model_name, history in histories.items():
        class_ious[model_name] = np.stack([t.numpy() for t in history['class_iou']])
        class_ious[model_name] = np.delete(class_ious[model_name], ignore_index, axis=-1)
        class_ious[model_name] = {label_name: class_ious[model_name][:, i] for i, label_name in enumerate(label_names)}

    plot(losses, 'Epoch', 'Value', 'Loss')
    plot(mious, 'Epoch', 'Value', 'MIoU')
    plot(accuracies, 'Epoch', 'Value', 'Accuracy')

    for model_name, class_iou in class_ious.items():
        plot(class_iou, 'Epoch', 'Value', f'{model_name} Class IoU')

        # TODO: Add confusion matrix of the best epoch
        # TODO: Add gradient flow of the last epoch


if __name__ == '__main__':
    main()
