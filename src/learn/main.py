import numpy as np
import os
import logging

import torch
import wandb
from omegaconf import DictConfig

from src.datasets import SemanticDataset, SelectionDataset
from .trainer import Trainer, ActiveTrainer

log = logging.getLogger(__name__)


def train_semantic_model(cfg: DictConfig, device: torch.device):
    """ Train a semantic segmentation model.

    :param cfg: The configuration of the project.
    :param device: The device to train on.
    """

    resume = cfg.resume if 'resume' in cfg else False

    project_name = f'Semantic Model Training Dev'
    group_name = f'{cfg.model.architecture} {cfg.ds.name}'
    model_path = os.path.join(cfg.path.models, 'semantic', f'{cfg.model.architecture}_{cfg.ds.name}')

    run = wandb.init(project=project_name, group=group_name, resume=resume)

    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, init=bool(~resume))
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size, init=bool(~resume))

    log.info(f'Loaded train dataset: \n{train_ds}')
    log.info(f'Loaded val dataset: \n{val_ds}')

    trainer = Trainer(cfg, train_ds, val_ds, device, model_path, resume)
    trainer.train()


def train_semantic_model_active(cfg: DictConfig, device: torch.device):
    """ Train a semantic segmentation model with active learning.

    :param cfg: The configuration of the project.
    :param device: The device to train on.
    """
    resume = cfg.resume if 'resume' in cfg else False

    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train',
                               size=cfg.train.dataset_size, active_mode=True, resume=resume)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val',
                             size=cfg.train.dataset_size, active_mode=True, resume=resume)
    sel_ds = SelectionDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)

    log.info(f'Loaded train dataset: \n{train_ds}')
    log.info(f'Loaded val dataset: \n{val_ds}')

    method = 'entropy_voxels'
    model_path = os.path.join(cfg.path.models, 'active_semantic',
                              f'{cfg.model.architecture}_{cfg.ds.name}_{method}.pt')

    trainer = ActiveTrainer(cfg, train_ds, val_ds, sel_ds, device, model_path, method, resume)
    trainer.train()
