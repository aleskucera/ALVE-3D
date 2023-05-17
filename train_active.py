#!/usr/bin/env python
import logging

import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.io import set_paths
from src.learn.active import train_model_active, create_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    if 'device' in cfg:
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.option == 'train':
        train_model_active(cfg, device)
    elif cfg.option == 'create_seed':
        create_seed(cfg, device)
    else:
        raise ValueError(f'Option "{cfg.option}" is not supported')


if __name__ == '__main__':
    main()
