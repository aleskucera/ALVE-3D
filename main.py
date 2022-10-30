#!/usr/bin/env python

import time
import atexit
import logging

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src import train_model, test_model, set_paths, start_tensorboard, \
    terminate_tensorboard, supervise_remote, check_value

log = logging.getLogger(__name__)


@atexit.register
def exit_function():
    """ Terminate all running tensorboard processes when the program exits. """
    terminate_tensorboard()
    log.info('Terminated all running tensorboard processes.')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for training and testing the model. Configurations are loaded from the conf/config.yaml file.
    You can change the action by changing the action variable or running the following command in the terminal:

        python main.py action=train

    The data can be visualized in tensorboard at the url http://localhost:6006.

    :param cfg: Configuration object. The arguments are loaded automatically by Hydra.
    :return: None
    """

    # Set paths to absolute paths and update the output directory
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    # Check if the configuration is valid
    log.info(f"\nStarting a program with the following configuration: \
                    \n\t - node: {cfg.node} \
                    \n\t - action: {cfg.action} \
                    \n\t - connection: {cfg.connection}")

    check_value(cfg.node, ['master', 'slave'])
    check_value(cfg.action, ['train', 'test'])
    check_value(cfg.connection, ['local', 'remote'])

    if cfg.node == 'master':
        start_tensorboard(cfg.path.output)
        time.sleep(5)

        if cfg.connection == 'remote':
            supervise_remote(cfg)
        elif cfg.connection == 'local':
            run_action(cfg)
        else:
            log.error(f'The connection "{cfg.connection}" is not supported')

        input("\nPress ENTER to exit\n")

    elif cfg.node == 'slave':
        run_action(cfg)

    log.info('Exiting program')


def run_action(cfg: DictConfig) -> None:
    if cfg.action == 'train':
        train_model(cfg)
    elif cfg.action == 'test':
        test_model(cfg)
    else:
        log.error(f'The action "{cfg.action}" is not supported')


if __name__ == '__main__':
    main()
