#!/usr/bin/env python

import time
import atexit
import logging

import wandb
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src import train_model, train_model_active, test_model, set_paths, start_tensorboard, \
    terminate_tensorboard

log = logging.getLogger(__name__)


@atexit.register
def exit_function():
    """ Terminate all running tensorboard processes when the program exits. """
    terminate_tensorboard()
    log.info('Terminated all running tensorboard processes.')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set paths to absolute paths and update the output directory
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting action: {cfg.action}')

    start_tensorboard(cfg.path.output)
    time.sleep(5)
    with wandb.init(project='ALVE-3D'):
        if cfg.action == 'train':
            train_model(cfg)
        elif cfg.action == 'train_active':
            train_model_active(cfg)
        elif cfg.action == 'test':
            test_model(cfg)
        else:
            log.error(f'The action "{cfg.action}" is not supported')

    input("\nPress ENTER to exit\n")

    log.info('Exiting program')


if __name__ == '__main__':
    main()
