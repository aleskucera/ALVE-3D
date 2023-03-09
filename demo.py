#!/usr/bin/env python
import logging

import wandb
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src import SemanticDataset, LaserScan, ScanVis, visualize_superpoints, \
    create_config, KITTI360Converter

from src.utils.io import set_paths
from src.dataset.dataset2 import ActiveDataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'config':
        show_hydra_config(cfg)
    elif cfg.action == 'visualize_dataset':
        visualize_dataset(cfg)
    elif cfg.action == 'log_sequence':
        log_sequence(cfg)
    elif cfg.action == 'dataset_statistics':
        dataset_statistics(cfg)
    elif cfg.action == 'visualize_kitti360_conversion':
        visualize_kitti360_conversion(cfg)
    elif cfg.action == 'create_kitti360_config':
        create_config()
    elif cfg.action == 'superpoints':
        log_superpoints(cfg)
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


def visualize_dataset(cfg: DictConfig) -> None:
    """ Show how to use SemanticDataset.
    :param cfg: Configuration object.
    """

    split = 'train'
    size = None
    sequences = [0]
    indices = None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, cfg=cfg.ds,
                              split=split, sequences=sequences, indices=indices, size=size)

    # Create scan object
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # Visualizer
    vis = ScanVis(scan=scan, scans=dataset.scans, labels=dataset.labels, raw_cloud=True, instances=False)
    vis.run()


def log_sequence(cfg: DictConfig) -> None:
    """ Log sample from dataset sequence to Weights & Biases.

    :param cfg: Configuration object.
    """

    # sequence = cfg.sequence

    # train_ds = SemanticDataset(dataset_path=cfg.ds.path, split='train', sequences=[sequence], cfg=cfg.ds)
    # val_ds = SemanticDataset(dataset_path=cfg.ds.path, split='val', sequences=[sequence], cfg=cfg.ds)
    #
    # scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)
    #
    # if len(train_ds) > 0:
    #     with wandb.init(project='Sequence Visualization', group=cfg.ds.name, name=f'Sequence {sequence} - train'):
    #         _log_sequence(train_ds, scan)
    # else:
    #     log.info(f'Train dataset for sequence {sequence} is empty.')
    #
    # if len(val_ds) > 0:
    #     with wandb.init(project='Sequence Visualization', group=cfg.ds.name, name=f'Sequence {sequence} - val'):
    #         _log_sequence(val_ds, scan)
    # else:
    #     log.info(f'Validation dataset for sequence {sequence} is empty.')

    dataset = ActiveDataset(cfg.ds.path, cfg.ds, 'train')
    print(dataset)

    with wandb.init(project='Test new dataset'):
        for i in [1, 10, 36, 150]:
            proj_image, proj_label = dataset[i]
            wandb.log({'Projection': wandb.Image(proj_image[:, :, 2:]),
                       'Projection Label': wandb.Image(proj_label)})


def _log_sequence(dataset, scan):
    i = np.random.randint(0, len(dataset))
    scan.open_scan(dataset.scans[i])
    scan.open_label(dataset.labels[i])

    cloud = np.concatenate([scan.points, scan.color * 255], axis=1)
    label = np.concatenate([scan.points, scan.sem_label_color * 255], axis=1)

    wandb.log({'Point Cloud': wandb.Object3D(cloud),
               'Point Cloud Label': wandb.Object3D(label),
               'Projection': wandb.Image(scan.proj_color),
               'Projection Label': wandb.Image(scan.proj_sem_color)})

    print(f'\nLogged scan: {dataset.scans[i]} \n'
          f'Logged label: {dataset.labels[i]} \n')


def dataset_statistics(cfg: DictConfig):
    dataset = SemanticDataset(dataset_path=cfg.ds.path, split='train', cfg=cfg.ds)
    dataset.calculate_statistics()


def visualize_kitti360_conversion(cfg: DictConfig):
    """ Visualize KITTI360 conversion.
    :param cfg: Configuration object.
    """

    converter = KITTI360Converter(cfg)
    converter.visualize()


def log_superpoints(cfg: DictConfig):
    visualize_superpoints(cfg)


if __name__ == '__main__':
    main()
