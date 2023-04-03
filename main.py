#!/usr/bin/env python

import logging

import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.learn import train_model_full, select_first_voxels, select_voxels, train_model
from src.utils import set_paths
from src.kitti360 import KITTI360Converter

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set paths to absolute paths and update the output directory
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    if 'device' in cfg:
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f'Starting action: {cfg.action} using device {device}')

    if cfg.action == 'train':
        train_model_full(cfg, device)
    elif cfg.action == 'convert_kitti360':
        converter = KITTI360Converter(cfg)
        # converter.create_global_clouds()
        # converter.convert()
    elif cfg.action == 'create_kitti360_superpoints':
        converter = KITTI360Converter(cfg)
        # converter.create_global_clouds()
        converter.create_superpoints(device)
    elif cfg.action == 'select_first_voxels':
        select_first_voxels(cfg, device)
    elif cfg.action == 'select_voxels':
        select_voxels(cfg, device)
    elif cfg.action == 'train_model':
        train_model(cfg, device)
    else:
        log.error(f'The action "{cfg.action}" is not supported')

    log.info(f'Finished action: {cfg.action}')


if __name__ == '__main__':
    main()
