#!/usr/bin/env python
import logging

import wandb
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils import set_paths, visualize_global_cloud
from src.datasets import SemanticDataset
from src.kitti360 import KITTI360Converter, create_kitti360_config
from src.laserscan import LaserScan, ScanVis

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'config':
        show_hydra_config(cfg)
    elif cfg.action == 'visualize_dataset':
        visualize_dataset(cfg)
    elif cfg.action == 'visualize_sequence':
        visualize_sequence(cfg)
    elif cfg.action == 'log_sequence':
        log_sequence(cfg)
    elif cfg.action == 'create_kitti360_config':
        create_kitti360_config()
    elif cfg.action == 'visualize_kitti360_conversion':
        visualize_kitti360_conversion(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def show_hydra_config(cfg: DictConfig) -> None:
    """ Show how to use Hydra for configuration management.
    :param cfg: Configuration object.
    """
    print('\nThis project uses Hydra for configuration management.')
    print(f'Configuration object type is: {type(cfg)}')
    print(f'Configuration content is separated into groups:')
    for group in cfg.keys():
        print(f'\t{group}')

    print('\nPaths dynamically generated to DictConfig object:')
    for name, path in cfg.path.items():
        print(f'\t{name}: {path}')
    print('')


def visualize_dataset(cfg: DictConfig):
    """ Show how to use SemanticDataset.

    :param cfg: Configuration object.
    """

    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = cfg.sequences if 'sequences' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, cfg=cfg.ds, split=split, size=size, sequences=sequences)

    # Create scan object
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # Visualizer
    vis = ScanVis(scan=scan, scans=dataset.scan_files, labels=dataset.label_files, raw_cloud=True, instances=False)
    vis.run()


def visualize_sequence(cfg: DictConfig) -> None:
    """ Show how to use SemanticDataset.
    :param cfg: Configuration object.
    """

    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else [3]

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, cfg=cfg.ds,
                              split=split, size=size, sequences=sequences)

    # Create scan object
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # Load the sequence
    points, colors, poses = [], [], []
    for i in tqdm(range(len(dataset)), desc='Loading sequence'):
        scan.open_scan(dataset.scan_files[i])
        scan.open_label(dataset.label_files[i])
        pose = dataset.poses[i]

        points.append(scan.points)
        colors.append(scan.sem_label_color)
        poses.append(pose)

    visualize_global_cloud(points, colors, poses, step=5, voxel_size=0.2)


def log_sequence(cfg: DictConfig) -> None:
    """ Log sample from dataset sequence to Weights & Biases.

    :param cfg: Configuration object.
    """

    sequence = cfg.sequence if 'sequence' in cfg else 3

    train_ds = SemanticDataset(dataset_path=cfg.ds.path, split='train', sequences=[sequence], cfg=cfg.ds)
    val_ds = SemanticDataset(dataset_path=cfg.ds.path, split='val', sequences=[sequence], cfg=cfg.ds)

    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    if len(train_ds) > 0:
        with wandb.init(project='Sequence Visualization', group=cfg.ds.name, name=f'Sequence {sequence} - train'):
            _log_sequence(train_ds, scan)
    else:
        log.info(f'Train dataset for sequence {sequence} is empty.')

    if len(val_ds) > 0:
        with wandb.init(project='Sequence Visualization', group=cfg.ds.name, name=f'Sequence {sequence} - val'):
            _log_sequence(val_ds, scan)
    else:
        log.info(f'Validation dataset for sequence {sequence} is empty.')


def _log_sequence(dataset, scan):
    i = np.random.randint(0, len(dataset))
    scan.open_scan(dataset.scan_files[i])
    scan.open_label(dataset.label_files[i])

    cloud = np.concatenate([scan.points, scan.color * 255], axis=1)
    label = np.concatenate([scan.points, scan.sem_label_color * 255], axis=1)

    wandb.log({'Point Cloud': wandb.Object3D(cloud),
               'Point Cloud Label': wandb.Object3D(label),
               'Projection': wandb.Image(scan.proj_color),
               'Projection Label': wandb.Image(scan.proj_sem_color)})

    log.info(f'Logged scan: {dataset.scan_files[i]}')
    log.info(f'Logged label: {dataset.label_files[i]}')


def visualize_kitti360_conversion(cfg: DictConfig):
    """ Visualize KITTI360 conversion.
    :param cfg: Configuration object.
    """

    converter = KITTI360Converter(cfg)
    converter.visualize()


if __name__ == '__main__':
    main()
