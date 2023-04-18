#!/usr/bin/env python

import logging

import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils import set_paths
from src.kitti360 import KITTI360Converter
from src.semantickitti import SemanticKITTIConverter
from src.learn import train_semantic_model, train_partition_model, \
    train_semantic_active, train_partition_active, select_voxels

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    if 'device' in cfg:
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f'Starting action: {cfg.action} using device {device}')

    if cfg.action == 'train_semantic':
        train_semantic_model(cfg, device)
    elif cfg.action == 'train_partition':
        train_partition_model(cfg, device)
    elif cfg.action == 'train_semantic_active':
        train_semantic_active(cfg, device)
    elif cfg.action == 'train_partition_active':
        train_partition_active(cfg, device)
    elif cfg.action == 'select_voxels':
        select_voxels(cfg, device)
    # elif cfg.action == 'select_first_voxels':
    #     select_first_voxels(cfg, device)
    elif cfg.action == 'convert_dataset':
        if cfg.ds.name == 'KITTI-360':
            converter = KITTI360Converter(cfg)
        elif cfg.ds.name == 'SemanticKITTI':
            converter = SemanticKITTIConverter(cfg)
        else:
            raise NotImplementedError(f'The dataset "{cfg.ds.name}" is not supported')
        converter.convert()
    else:
        log.error(f'The action "{cfg.action}" is not supported')

    log.info(f'Finished action: {cfg.action}')


if __name__ == '__main__':
    main()
