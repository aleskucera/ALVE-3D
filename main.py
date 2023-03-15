#!/usr/bin/env python

import logging

import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.learn import train_semantic_model, train_semantic_model_active
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
        train_semantic_model(cfg, device)
    elif cfg.action == 'train_active':
        train_semantic_model_active(cfg, device)
    elif cfg.action == 'convert_kitti360':
        converter = KITTI360Converter(cfg)
        converter.create_global_clouds()
        converter.convert()
    else:
        log.error(f'The action "{cfg.action}" is not supported')

    log.info(f'Finished action: {cfg.action}')


if __name__ == '__main__':
    main()
