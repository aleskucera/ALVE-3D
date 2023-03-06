import logging

import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .parser import SemanticParser
from .basetrainer import BaseTrainer

log = logging.getLogger(__name__)


class SemanticTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, device: torch.device):
        self.wandb_id = wandb.util.generate_id()
        self.project_name = f'{cfg.model.architecture}_{cfg.ds.name}_{cfg.action}'
        with wandb.init(project=self.project_name, id=self.wandb_id):
            super().__init__(cfg, train_ds, val_ds, SemanticParser(), device)

    def train(self, epochs: int):
        with wandb.init(project=self.project_name, id=self.wandb_id):
            for epoch in range(epochs):
                self.train_epoch(epoch, validate=True)
