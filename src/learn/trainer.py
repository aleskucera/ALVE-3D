import logging

import torch
import wandb
from tqdm import tqdm
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .logger import get_logger
from src.losses import get_loss
from src.models import get_model
from src.datasets import get_parser

log = logging.getLogger(__name__)


class BaseTrainer(object):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset,
                 device: torch.device, state: dict = None):
        self.cfg = cfg
        self.device = device
        self.val_ds = val_ds
        self.train_ds = train_ds

        self.batch_size = cfg.train.batch_size
        self.num_workers = torch.cuda.device_count() * 4 if device.type == 'cuda' else 4

        self.model = get_model(cfg, device)
        self.loss_fn = get_loss(cfg.model.type, device)
        self.parser = get_parser(cfg.model.type, device)
        self.logger = get_logger(cfg.model.type, cfg.ds.num_classes, cfg.ds.labels_train, device, cfg.ds.ignore_index)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.learning_rate)

        self.epoch = 0

        if state is not None:
            self.load_state(state)

    def train(self):
        raise NotImplementedError

    def train_epoch(self, validate: bool = True):
        """ Train the model for one epoch. If validate is True, the model will be validated after the epoch.

        :param validate: Whether to validate the model after the epoch.
        :return: The logger history. Can be used to save the model state and then resume training from that point.
        """
        self.model.train()
        loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Training epoch number {self.epoch}')):
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Load the batch
            inputs, targets = self.parser.parse_batch(batch)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Update the loss and metrics of the current batch
            with torch.no_grad():
                self.logger.update(loss.item(), outputs, targets)

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
        loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f'Validation epoch number {self.epoch}')):
                # Load the batch
                inputs, targets = self.parser.parse_batch(batch)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # Update the loss and metrics of the current batch
                self.logger.update(loss.item(), outputs, targets)

        # Calculate the loss and metrics of the current epoch and log them, then return the history
        self.logger.log_val(self.epoch)

    def save_state(self, history: dict):
        log.info(f'Saving model state ...')

        # Create state dictionary and save it
        self.model.eval()
        state_dict = {'epoch': self.epoch,
                      'history': history,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(state_dict, 'state.pt')

        # Create metadata for W&B artifact
        metadata = {'epoch': self.epoch}
        for key, value in history.items():
            metadata[key] = value

        # Create W&B artifact and log it
        artifact = wandb.Artifact('state', type='model', metadata=metadata,
                                  description='Model state with optimizer state and metric history for each epoch.')
        artifact.add_file('base_state.pt')
        wandb.run.log_artifact(artifact)

    def load_state(self, state: dict):
        self.epoch = state['epoch']
        self.logger.load_history(state['history'])
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])


class Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, device: torch.device,
                 state: dict = None):
        super().__init__(cfg, train_ds, val_ds, device, state)

    def train(self):
        self.epoch = 0
        self.logger.reset()
        class_distribution, class_progress, labeled_ratio = self.train_ds.get_statistics()
        self.logger.log_dataset_statistics(class_distribution, class_progress, labeled_ratio)
        while not self.logger.miou_converged:
            self.train_epoch(validate=True)
            if self.logger.miou_improved():
                self.save_state(self.logger.history)
            self.epoch += 1
