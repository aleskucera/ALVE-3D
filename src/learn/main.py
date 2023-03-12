import os
import logging

import torch
import wandb
from omegaconf import DictConfig

from src.datasets import SemanticDataset
from .trainer import Trainer, ActiveTrainer

log = logging.getLogger(__name__)


def train_semantic_model(cfg: DictConfig, device: torch.device):
    """ Train a semantic segmentation model.

    :param cfg: The configuration of the project.
    :param device: The device to train on.
    """

    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size)

    project_name = f'Semantic Model Training'
    group_name = f'{cfg.model.architecture} {cfg.ds.name}'
    model_name = f'{cfg.model.architecture}_{cfg.ds.name}'
    output_model_dir = os.path.join(cfg.path.models, 'semantic')

    with wandb.init(project=project_name, group=group_name):
        trainer = Trainer(cfg, train_ds, val_ds, device, model_name, output_model_dir)
        trainer.train(cfg.train.epochs)


def train_semantic_model_active(cfg: DictConfig, device: torch.device):
    """ Train a semantic segmentation model with active learning.

    :param cfg: The configuration of the project.
    :param device: The device to train on.
    """

    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, active_mode=True)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size, active_mode=True)

    project_name = f'Active Semantic Model Training'
    group_name = f'{cfg.model.architecture} {cfg.ds.name}'
    model_name = f'{cfg.model.architecture}_{cfg.ds.name}'
    output_model_dir = os.path.join(cfg.path.models, 'semantic_active')

    with wandb.init(project=project_name, group=group_name):
        trainer = ActiveTrainer(cfg, train_ds, val_ds, device, model_name, output_model_dir, 'random_voxels')
        trainer.train(cfg.train.epochs)


def train_partition_model(cfg: DictConfig, device: torch.device):
    """ Train a transformation model for partitioning a point cloud into a superpoints.

    :param cfg: The configuration of the project.
    :param device: The device to train on.
    """

    raise NotImplementedError
