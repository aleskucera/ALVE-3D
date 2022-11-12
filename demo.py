#!/usr/bin/env python

import os
import time
import math
import random
import atexit
import logging

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

from src import SemanticDataset, set_paths, start_tensorboard, \
    terminate_tensorboard, SemLaserScan, ScanVis, Selector

log = logging.getLogger(__name__)


@atexit.register
def exit_function():
    """ Terminate all running tensorboard processes when the program exits. """
    terminate_tensorboard()
    log.info('Terminated all running tensorboard processes.')


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
    elif cfg.action == 'monitoring':
        start_tensorboard(cfg.path.output)
        time.sleep(5)
        computational_simulation(cfg)
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
    size = 1000
    sequences = [0, 1, 2, 3, 4]
    indices = random.sample(range(1, size), 10)

    # create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path,
                              sequences=sequences, cfg=cfg.ds, indices=indices, size=size)

    # create semantic laser scan
    scan = SemLaserScan(colorize=True, sem_color_dict=cfg.ds.color_map)

    # create scan visualizer
    vis = ScanVis(scan=scan, scan_names=dataset.points, label_names=dataset.labels, semantics=True)
    vis.run()


def computational_simulation(cfg: DictConfig) -> None:
    writer = SummaryWriter(cfg.path.output)
    for i in range(500):
        x = i / 10
        a = random.random()
        writer.add_scalar('Sin(x)', a * math.sin(x), i)
        writer.add_scalar('Cos(x)', a * math.cos(x), i)
        log.info(f'Iteration {i} completed')
        time.sleep(0.5)


def select_indices(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=100)

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=os.cpu_count() // 2)

    model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
    model = torch.load(model_path).to(device)

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
