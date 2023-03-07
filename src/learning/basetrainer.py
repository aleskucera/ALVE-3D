import logging

import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from .loss import get_loss
from .logger import get_logger
from .parser import BaseParser
from src.model import get_model

log = logging.getLogger(__name__)


class BaseTrainer(object):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, parser: BaseParser,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.parser = parser

        batch_size, num_workers = cfg.train.batch_size, self._get_num_workers(device)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.model = get_model(cfg, device)
        self.loss_fn = get_loss(cfg.model.type, device)
        self.logger = get_logger(cfg.model.type, cfg.ds.num_classes, cfg.ds.labels_train, device, cfg.ds.ignore_index)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.learning_rate)

    def train(self, epochs: int):
        raise NotImplementedError

    def train_epoch(self, epoch: int, validate: bool = True) -> dict:
        self.model.train()

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()

            inputs, targets = self.parser.parse_data(batch, self.device)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self.optimizer.step()

            self.logger.update(loss.item(), outputs, targets)
        self.logger.log_train(epoch)

        if validate:
            return self.validate(epoch)
        else:
            return {}

    def validate(self, epoch: int) -> dict:
        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            inputs, targets = self.parser.parse_data(batch, self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            self.logger.update(loss.item(), outputs, targets)
        return self.logger.log_val(epoch)

    @staticmethod
    def _get_num_workers(device: torch.device) -> int:
        if device.type == 'cuda':
            return torch.cuda.device_count() * 4
        else:
            return 4
