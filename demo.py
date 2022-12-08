#!/usr/bin/env python

import os
import random
import logging

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

from src import SemanticDataset, set_paths, LaserScan, ScanVis, Selector, create_model

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
    # size = 200
    size = None
    sequences = None
    indices = random.sample(range(1, 100), 10)

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

    test_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=100)

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=os.cpu_count() // 2)

    model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
    model = create_model(cfg)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    tester = Selector(model=model, loader=test_loader, device=device)
    entropies = tester.calculate_entropies()

    print(f'\nEntropies shape: {entropies.shape}')

    entropies = entropies[:10]
    print(f'\nFirst 10 entropies [entropy value, index in dataset]: \n{entropies}')

    print(f'\nCorresponding scan files: ')
    for _, i in entropies:
        print(f'\t{test_ds.points[int(i)]}')
    print('')


if __name__ == '__main__':
    main()
