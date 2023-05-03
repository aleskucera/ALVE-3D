#!/usr/bin/env python
import atexit
import logging

import torch
import hydra
import wandb
import omegaconf
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.io import set_paths
from src.selection import select_voxels
from src.kitti360 import KITTI360Converter
from src.utils.experiment import Experiment
from src.semantickitti import SemanticKITTIConverter
from src.superpoints import create_superpoints, compute_redal_features
from src.learn import train_model, train_model_active, train_semantickitti_original

log = logging.getLogger(__name__)


def error_alert(experiment: Experiment):
    log.error(f'Experiment failed: {experiment}')
    wandb.alert(
        title='Experiment failed',
        text=f'Experiment info: {experiment}',
        level=wandb.AlertLevel.ERROR
    )


def success_alert(experiment: Experiment):
    log.info(f'Experiment finished successfully: {experiment}')
    wandb.alert(
        title='Experiment finished successfully',
        text=f'Experiment info: {experiment}',
        level=wandb.AlertLevel.INFO
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    if 'device' in cfg:
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f'Starting action: {cfg.action} using device {device}')

    experiment = Experiment(cfg)
    atexit.register(error_alert, experiment)
    dict_config = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(project=experiment.project, group=experiment.group,
                    name=experiment.name, config=dict_config, job_type=experiment.job_type):
        if cfg.action == 'train_model':
            train_model(cfg, experiment, device)
        elif cfg.action == 'train_semantickitti_original':
            train_semantickitti_original(cfg, experiment, device)
        elif cfg.action == 'train_model_active':
            train_model_active(cfg, experiment, device)
        elif cfg.action == 'select_voxels':
            select_voxels(cfg, experiment, device)
        elif cfg.action == 'convert_dataset':
            if cfg.ds.name == 'KITTI-360':
                converter = KITTI360Converter(cfg)
            elif cfg.ds.name == 'SemanticKITTI':
                converter = SemanticKITTIConverter(cfg)
            else:
                raise NotImplementedError(f'The dataset "{cfg.ds.name}" is not supported')
            converter.convert()
        elif cfg.action == 'create_superpoints':
            create_superpoints(cfg)
        elif cfg.action == 'compute_redal_features':
            compute_redal_features(cfg)
        else:
            log.error(f'The action "{cfg.action}" is not supported')

        success_alert(experiment)
        atexit.unregister(error_alert)

    log.info(f'Finished action: {cfg.action}')


if __name__ == '__main__':
    main()
