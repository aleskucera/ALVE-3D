import logging

import torch
import wandb
from omegaconf import DictConfig

from src.datasets import SemanticDataset
from src.learning import Trainer, ActiveTrainer

log = logging.getLogger(__name__)

gpu_count = torch.cuda.device_count()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log.info(f'Using device {device}')


def train_model(cfg: DictConfig):
    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size)

    project_name = f'{cfg.model.architecture}_{cfg.ds.name}_{cfg.action}'
    model_name = f'{cfg.model.architecture}_{cfg.ds.name}'

    with wandb.init(project=project_name):
        trainer = Trainer(cfg, train_ds, val_ds, device, model_name)
        trainer.train(cfg.train.epochs)


def train_model_active(cfg: DictConfig):
    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size)

    project_name = f'{cfg.model.architecture}_{cfg.ds.name}_{cfg.action}'
    model_name = f'{cfg.model.architecture}_{cfg.ds.name}'

    with wandb.init(project=project_name):
        trainer = ActiveTrainer(cfg, train_ds, val_ds, device, model_name)
        trainer.train(cfg.train.epochs)
