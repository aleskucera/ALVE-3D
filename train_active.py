#!/usr/bin/env python
import logging

import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.io import set_paths
from src.learn import train_model, train_model_active, train_semantickitti_original

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    if 'device' in cfg:
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f'Starting active learning action: {cfg.action} using device {device}')

    if cfg.action == 'full_experiment':
        train_model_active(cfg, device)
    # elif cfg.action == 'train':
    #     train(cfg, device)
    # elif cfg.action == 'select':
    #     select(cfg, device)
    else:
        raise ValueError(f'Action "{cfg.action}" is not supported')

    log.info(f'Finished active learning action: {cfg.action}')


if __name__ == '__main__':
    main()
