#!/usr/bin/env python

import os
import logging

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

from src import SemanticDataset, set_paths, LaserScan, ScanVis, Selector, SalsaNext

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'paths':
        show_paths(cfg)
    elif cfg.action == 'dataset':
        show_dataset(cfg)
    elif cfg.action == 'select_indices':
        select_indices(cfg)
    elif cfg.action == 'common_points':
        common_points(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def show_paths(cfg: DictConfig) -> None:
    print('\nPaths dynamically generated to DictConfig object:')
    for name, path in cfg.path.items():
        print(f'\t{name}: {path}')
    print('')


def show_dataset(cfg: DictConfig) -> None:
    # dataset attributes
    size = None
    sequences = None
    indices = None

    # create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=sequences, cfg=cfg.ds,
                              split='train', indices=indices, size=size)

    # create semantic laser scan
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # create scan visualizer
    vis = ScanVis(scan=scan, scans=dataset.points, labels=dataset.labels, raw_cloud=True, instances=True)
    vis.run()


def select_indices(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid')

    test_loader = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=4)

    model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
    model = SalsaNext(cfg.ds.num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    selector = Selector(model=model, loader=test_loader, device=device)
    # entropies, indices = tester.calculate_entropies()
    #
    # print(f'\nFirst 10 entropies: \n{entropies[:10]}')
    # print(f'\nFirst 10 indices: \n{indices[:10]}')
    # print(f'\nFirst 10 ious: \n{ious[:10]}')


def common_points(cfg: DictConfig):
    size = None
    sequences = None
    indices = None
    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=sequences, cfg=cfg.ds,
                              split='train', indices=indices, size=size)

    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # create scan visualizer
    vis = ScanVis(scan=scan, scans=dataset.points, labels=dataset.labels, raw_cloud=True, instances=True)
    vis.run()


if __name__ == '__main__':
    main()
