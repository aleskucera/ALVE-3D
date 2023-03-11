import logging

import torch
import wandb
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from .loss import get_loss
from .logger import get_logger
from src.models import get_model
from src.datasets import get_parser
from src.active_selectors import get_selector
from src.losses import LovaszSoftmax

log = logging.getLogger(__name__)


class BaseTrainer(object):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, device: torch.device, model_name: str):
        self.cfg = cfg
        self.device = device
        self.val_ds = val_ds
        self.train_ds = train_ds
        self.model_name = model_name

        batch_size, num_workers = cfg.train.batch_size, self._get_num_workers(device)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.model = get_model(cfg, device)
        self.loss_fn = get_loss(cfg.model.type, device)
        self.parser = get_parser(cfg.model.type, device)
        self.logger = get_logger(cfg.model.type, cfg.ds.num_classes, cfg.ds.labels, device, cfg.ds.ignore_index)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.learning_rate)

    def train(self, epochs: int):
        raise NotImplementedError

    def train_epoch(self, epoch: int, validate: bool = True) -> dict:
        self.model.train()

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()

            inputs, targets = self.parser.parse_batch(batch)

            print(f'inputs shape: {inputs.shape}')
            print(f'targets shape: {targets.shape}')

            outputs = self.model(inputs)

            print(f'outputs shape: {outputs.shape}')
            loss_fn = LovaszSoftmax(ignore=0)
            loss = loss_fn(outputs, targets)
            # loss = self.loss_fn(outputs, targets)
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

    def save_state(self, epoch: int, results: dict):
        self.model.eval()
        save_dict = {'epoch': epoch,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict()}

        for key, value in results.items():
            save_dict[key] = value

        torch.save(save_dict, self.model_name)


class Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, device: torch.device, model_name: str,
                 main_metric: str = 'iou'):
        super().__init__(cfg, train_ds, val_ds, device, model_name)
        self.max_main_metric = 0
        self.main_metric = main_metric

    def train(self, epochs: int):
        for epoch in range(epochs):
            results = self.train_epoch(epoch, validate=True)

            if results[self.main_metric] > self.max_main_metric:
                self.max_main_metric = results[self.main_metric]
                self.save_state(epoch, results)


class ActiveTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, device: torch.device, model_name: str):
        super().__init__(cfg, train_ds, val_ds, device, model_name)
        self.max_iou = 0
        self.selector = get_selector('std', self.train_ds.path, self.train_ds.sequences, device)

    def train(self, epochs: int):
        while not self.selector.is_finished():
            self.selector.select(self.train_ds, self.model)
            self.train_ds.update()
            break
