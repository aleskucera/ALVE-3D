import logging
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .logger import get_logger
from src.losses import get_loss
from src.models import get_model
from src.datasets import Dataset, get_parser
from src.utils.log import log_model, log_history

log = logging.getLogger(__name__)


class BaseTrainer(object):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset,
                 device: torch.device, weights: np.ndarray = None):
        self.cfg = cfg
        self.device = device
        self.val_ds = val_ds
        self.train_ds = train_ds
        weights = torch.from_numpy(weights).type(torch.float32) if weights is not None else None

        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers if device.type == 'cuda' else 4

        self.model = get_model(cfg, device)
        self.loss_fn = get_loss(cfg.train.loss, weights, device)
        self.parser = get_parser(train_ds.parser_type, device)
        self.logger = get_logger(cfg.model.type, cfg.ds.num_classes, cfg.ds.labels_train, device, cfg.ds.ignore_index)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.learning_rate)

        self.epoch = 0
        self.min_epochs = cfg.train.min_epochs
        self.patience = cfg.train.patience

        # self.model_name = model_name if model_name is not None else 'model'
        # self.history_name = history_name if history_name is not None else 'history'
        #
        # if model is not None:
        #     self.load_model(model)

    @property
    def history(self):
        return self.logger.history

    def train(self):
        raise NotImplementedError

    def train_epoch(self, validate: bool = True):
        """ Train the model for one epoch. If validate is True, the model will be validated after the epoch.

        :param validate: Whether to validate the model after the epoch.
        :return: The logger history. Can be used to save the model state and then resume training from that point.
        """

        self.model.train()
        # self.train_ds.train_mode()
        self.train_ds.selection_mode = False
        loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Training epoch number {self.epoch}')):
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Load the batch
            inputs, targets = self.parser.parse_batch(batch)

            # Forward pass
            outputs = self.model(inputs)
            if isinstance(outputs, dict) or isinstance(outputs, OrderedDict):
                outputs = outputs['out']
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the loss and metrics of the current batch
            with torch.no_grad():
                self.logger.update(loss.item(), outputs, targets, self.model.named_parameters())

            # Update the weights
            self.optimizer.step()

        # Calculate the loss and metrics of the current epoch and log them
        with torch.no_grad():
            self.logger.log_train(self.epoch)

        # Validate the model
        if validate:
            self.validate()

    def validate(self):
        """ Validate the model.

        :return: The logger history. Can be used to save the model state and then resume training from that point.
        """

        self.model.eval()
        # self.val_ds.train_mode()
        self.val_ds.selection_mode = False
        loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f'Validation epoch number {self.epoch}')):
                # Load the batch
                inputs, targets = self.parser.parse_batch(batch)

                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, dict) or isinstance(outputs, OrderedDict):
                    outputs = outputs['out']
                loss = self.loss_fn(outputs, targets)

                # Update the loss and metrics of the current batch
                self.logger.update(loss.item(), outputs, targets)

        # Calculate the loss and metrics of the current epoch and log them, then return the history
        self.logger.log_val(self.epoch)

    def load_model(self, model: dict):
        self.model.load_state_dict(model['model_state_dict'])


class SemanticTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, device: torch.device,
                 weights: np.ndarray = None):
        super().__init__(cfg, train_ds, val_ds, device, weights)
        self.best_model = dict(state_dict=None, miou=0, epoch=0)
        self.max_epochs = cfg.train.max_epochs

    # def train(self):
    #     while not self.logger.miou_converged(self.min_epochs, self.patience):
    #         self.train_epoch(validate=True)
    #         if self.logger.miou_improved():
    #             log_model(model=self.model, history=self.logger.history,
    #                       epoch=self.epoch, model_name=self.model_name)
    #         self.epoch += 1
    #
    #     log_history(history=self.logger.history, history_name=self.history_name)

    def reset(self):
        self.epoch = 0
        self.logger.reset()
        self.model.load_state_dict(self.best_model['state_dict'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.train.learning_rate)
        
    def train(self):
        while not self.logger.miou_converged(self.min_epochs, self.patience):

            self.train_epoch(validate=True)

            if self.logger.miou_improved():
                self.best_model['state_dict'] = self.model.state_dict()
                self.best_model['miou'] = self.logger.history['miou_val'][-1]
                self.best_model['epoch'] = self.epoch

            self.epoch += 1

            if self.epoch >= self.max_epochs:
                break
