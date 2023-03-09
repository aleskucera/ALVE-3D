#!/usr/bin/env python

import logging

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src import train_model, test_model, train_partition
from src.utils import set_paths
from src.kitti360 import KITTI360Converter

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set paths to absolute paths and update the output directory
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting action: {cfg.action}')

    if cfg.action == 'train':
        train_model(cfg)
    elif cfg.action == 'test':
        test_model(cfg)
    elif cfg.action == 'train_partition':
        train_partition(cfg)
    elif cfg.action == 'convert_kitti360':
        converter = KITTI360Converter(cfg)
        converter.create_global_clouds()
        converter.convert()
    else:
        log.error(f'The action "{cfg.action}" is not supported')

    log.info(f'Finished action: {cfg.action}')


if __name__ == '__main__':
    main()
