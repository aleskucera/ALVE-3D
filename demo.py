#!/usr/bin/env python

import time
import math
import random
import atexit
import logging

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

from src import SemanticDataset, set_paths, \
    check_value, supervise_remote, start_tensorboard, \
    terminate_tensorboard, SemLaserScan, ScanVis

log = logging.getLogger(__name__)


@atexit.register
def exit_function():
    """ Terminate all running tensorboard processes when the program exits. """
    terminate_tensorboard()
    log.info('Terminated all running tensorboard processes.')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)
    log.info(f'Starting demo with action: {cfg.action}')
    check_value(cfg.node, ['master', 'slave'])
    check_value(cfg.action, ['paths', 'simulation', 'dataset'])

    if cfg.action == 'paths':
        show_paths(cfg)
    elif cfg.action == 'dataset':
        show_dataset(cfg)
    elif cfg.action == 'simulation':
        if cfg.node == 'master':
            start_tensorboard(cfg.path.output)
            time.sleep(10)
            log.info('Starting supervisor')
            supervise_remote(cfg)
            pass
        elif cfg.node == 'slave':
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
    dataset = SemanticDataset(dataset_path=cfg.path.kitti,
                              sequences=sequences, cfg=cfg.kitti, indices=indices, size=size)

    # create semantic laser scan
    scan = SemLaserScan(colorize=True, sem_color_dict=cfg.kitti.color_map)

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


if __name__ == '__main__':
    main()
