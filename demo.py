#!/usr/bin/env python
import os
import logging

import torch
import wandb
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.models import get_model
from src.datasets import SemanticDataset
from src.utils.cloud import visualize_cloud
from src.laserscan import LaserScan, ScanVis
from src.superpoints import partition_cloud, calculate_features, compute_color_discontinuity, compute_surface_variation
from src.utils.io import set_paths, ScanInterface, CloudInterface
from src.kitti360 import KITTI360Converter, create_kitti360_config
from src.utils.map import map_colors, colorize_values, colorize_instances
from src.utils.visualize import plot, bar_chart, grouped_bar_chart, plot_confusion_matrix

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'config_object':
        show_hydra_config(cfg)
    elif cfg.action == 'test':
        test(cfg)
    elif cfg.action == 'visualize_dataset_scans':
        visualize_dataset_scans(cfg)
    elif cfg.action == 'visualize_dataset_clouds':
        visualize_dataset_clouds(cfg)
    elif cfg.action == 'visualize_dataset_statistics':
        visualize_dataset_statistics(cfg)
    elif cfg.action == 'visualize_feature':
        visualize_feature(cfg)
    elif cfg.action == 'visualize_superpoints':
        visualize_superpoints(cfg)
    elif cfg.action == 'test_model':
        test_model(cfg)
    elif cfg.action == 'visualize_model_training':
        visualize_model_training(cfg)
    elif cfg.action == 'visualize_model_experiment':
        visualize_model_experiment(cfg)
    elif cfg.action == 'create_kitti360_config':
        create_kitti360_config()
    elif cfg.action == 'visualize_kitti360_conversion':
        visualize_kitti360_conversion(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def test(cfg: DictConfig):
    print(cfg)


def show_hydra_config(cfg: DictConfig) -> None:
    """ Show how to use Hydra for configuration management.
    :param cfg: Configuration object.
    """
    print('\nThis project uses Hydra for configuration management.')
    print(f'Configuration object type is: {type(cfg)}')
    print(f'Configuration content is separated into groups:')
    for group in cfg.keys():
        print(f'\t{group}')

    print(cfg.train.batch_size)

    print('\nPaths dynamically generated to DictConfig object:')
    for name, path in cfg.path.items():
        print(f'\t{name}: {path}')
    print('')


def test_model(cfg: DictConfig) -> None:
    artifact_path = cfg.artifact_path
    size = cfg.size if 'size' in cfg else None
    model_name = artifact_path.split('/')[-1].split(':')[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path,
                              project_name='demo', num_clouds=size)
    scan_interface = ScanInterface(dataset.project_name)

    laser_scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True,
                           H=cfg.ds.projection.H, W=cfg.ds.projection.W, fov_up=cfg.ds.projection.fov_up,
                           fov_down=cfg.ds.projection.fov_down)

    prediction_files = [path.replace('sequences', dataset.project_name) for path in dataset.scans]
    with wandb.init(project='demo', job_type='test_model'):
        # Load the model
        artifact_dir = wandb.use_artifact(artifact_path).download()
        model = torch.load(os.path.join(artifact_dir, f'{model_name}.pt'), map_location=device)
        model_state_dict = model['model_state_dict']
        model = get_model(cfg=cfg, device=device)
        model.load_state_dict(model_state_dict)

        # Save the predictions of the model on the validation set
        model.eval()
        with torch.no_grad():
            for i, scan_file in enumerate(dataset.scans):
                scan, label, _, _, _ = dataset[i]
                scan = torch.from_numpy(scan).to(device).unsqueeze(0)
                pred = model(scan)
                pred = pred.argmax(dim=1)
                pred = pred.cpu().numpy().squeeze()
                scan_interface.add_prediction(scan_file, pred)

        # Visualize the predictions
        vis = ScanVis(laser_scan=laser_scan, scans=dataset.scans, labels=dataset.scans,
                      predictions=prediction_files)
        vis.run()


def visualize_dataset_scans(cfg: DictConfig):
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)

    # Create scan object
    laser_scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True,
                           H=cfg.ds.projection.H, W=cfg.ds.projection.W, fov_up=cfg.ds.projection.fov_up,
                           fov_down=cfg.ds.projection.fov_down)

    # Visualizer
    vis = ScanVis(laser_scan=laser_scan, scans=dataset.scan_files, labels=dataset.scan_files,
                  raw_scan=True)
    vis.run()


def visualize_dataset_clouds(cfg: DictConfig):
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)
    cloud_interface = CloudInterface()

    for cloud_file in dataset.clouds:
        points = cloud_interface.read_points(cloud_file)
        labels = cloud_interface.read_labels(cloud_file)

        colors = map_colors(labels, cfg.ds.color_map_train)
        visualize_cloud(points, colors)


def visualize_dataset_statistics(cfg: DictConfig):
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    train_ds = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                               cfg=cfg.ds, split='train', num_clouds=size, sequences=sequences)
    val_ds = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                             cfg=cfg.ds, split='val', num_clouds=size, sequences=sequences)
    train_stats = train_ds.statistics
    val_stats = val_ds.statistics

    print('done')


def visualize_feature(cfg: DictConfig) -> None:
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    feature = cfg.feature if 'feature' in cfg else 'planarity'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)
    cloud_interface = CloudInterface()

    for cloud_file in dataset.clouds:
        points = cloud_interface.read_points(cloud_file)

        # Compute features
        feature = calculate_features(points)[feature]

        # Visualize feature
        log.info(f'{feature} max: {np.max(feature)}, {feature} min: {np.min(feature[feature != -1])}')
        feature_colors = colorize_values(feature, color_map='viridis', data_range=(0, np.max(feature)), ignore=(-1,))
        visualize_cloud(points, feature_colors)


def visualize_superpoints(cfg: DictConfig) -> None:
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)
    cloud_interface = CloudInterface()

    for cloud_file in dataset.clouds:
        points = cloud_interface.read_points(cloud_file)
        colors = cloud_interface.read_colors(cloud_file)
        edge_sources, edge_targets = cloud_interface.read_edges(cloud_file)

        components, component_map = partition_cloud(points=points, colors=colors,
                                                    edge_sources=edge_sources, edge_targets=edge_targets)

        superpoint_colors = colorize_instances(component_map)
        visualize_cloud(points, superpoint_colors)


def visualize_redal_features(cfg: DictConfig):
    pass


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


def visualize_loss_experiment(cfg: DictConfig) -> None:
    # TODO: Add train and validation loss
    # TODO: Add train and validation accuracy
    # TODO: Add train and validation MIoU

    raise NotImplementedError


def visualize_kitti360_conversion(cfg: DictConfig):
    """ Visualize KITTI360 conversion.
    :param cfg: Configuration object.
    """

    converter = KITTI360Converter(cfg)
    converter.visualize()


if __name__ == '__main__':
    main()
