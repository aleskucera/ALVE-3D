import os
import logging

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .state import State
from .utils import parse_data

log = logging.getLogger(__name__)


def train_requirements(func):
    def wrapper(self, *args, **kwargs):
        if self.train_loader is None:
            raise RuntimeError('Train loader is not set')
        if self.val_loader is None:
            raise RuntimeError('Validation loader is not set')
        if self.criterion is None:
            raise RuntimeError('Criterion is not set')
        if self.optimizer is None:
            raise RuntimeError('Optimizer is not set')
        log.info('All requirements for training are met, starting training')
        return func(self, *args, **kwargs)

    return wrapper


def test_requirements(func):
    def wrapper(self, *args, **kwargs):
        if self.test_loader is None:
            raise RuntimeError('Test loader is not set')
        log.info('All requirements for testing are met, starting testing')
        return func(self, *args, **kwargs)

    return wrapper


class Trainer:
    def __init__(self, model, metrics, device, log_path, criterion=None, optimizer=None,
                 train_loader=None, val_loader=None, test_loader=None,
                 patience=20, main_metric='MulticlassJaccardIndex'):

        self.model = model
        self.device = device

        # Train requirements
        self.patience = patience
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.train_loader = train_loader

        # Test requirements
        self.test_loader = test_loader

        self.state = State(writer=SummaryWriter(log_path),
                           metrics=metrics, logger=log,
                           main_metric=main_metric)

    @train_requirements
    def train(self, epochs: int, save_path: str) -> None:
        """ Train model for a number of epochs and save the best models to path """
        for epoch in range(epochs):

            # Train and validate
            self._train_epoch(epoch)
            self._val_epoch(epoch)

            # Save best models
            if self.state.main_metric_exceeded() and epoch > 10:
                log.info(f'Loss Decreasing, saving model to {save_path}')
                model_name = os.path.join(save_path, f'model_{epoch}.pth')
                self._save_model(os.path.join(save_path, model_name))

            # Early stopping
            if self.state.stagnant(patience=self.patience):
                log.info('Training Stagnant, stopping training')
                # break

    @test_requirements
    def test(self) -> dict:
        """ Test model on test set """
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                # Extract only inputs and labels
                image_batch, label_batch, _ = parse_data(data, self.device)

                # Forward pass (don't need loss for backward pass)
                output = self.model(image_batch)['out']
                self.state.accumulate_metrics(output, label_batch)

                # Compute metrics, log to tensorboard, console and reset
                self._compute_and_log(i, 'test')

        return self.state.metric_history

    def _train_epoch(self, epoch: int) -> None:
        """ Train model on training set """
        self.model.train()
        for data in tqdm(self.train_loader):
            # Extract only inputs and labels
            image_batch, label_batch, _ = parse_data(data, self.device)

            # Forward pass
            loss = self._forward_pass(image_batch, label_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Compute metrics, log to tensorboard, console and reset
        self._compute_and_log(epoch, 'train')

    def _val_epoch(self, epoch: int) -> None:
        """ Validate model on validation set """
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                # Extract only inputs and labels
                image_batch, label_batch, _ = parse_data(data, self.device)

                # Forward pass (don't need loss for backward pass)
                _ = self._forward_pass(image_batch, label_batch)

        # Compute metrics, log to tensorboard, console and reset
        self._compute_and_log(epoch, 'val')

    def _forward_pass(self, image_batch: torch.Tensor, label_batch: torch.Tensor) -> torch.Tensor:
        """ Forward pass through model, compute loss and accumulate metrics """
        output = self.model(image_batch)['out']
        loss = self.criterion(output, label_batch)
        self.state.accumulate_loss(loss.item())
        self.state.accumulate_metrics(output, label_batch)
        return loss

    def _compute_and_log(self, idx: int, phase: str) -> None:
        self.state.compute(phase)
        self.state.log(idx, phase)
        self.state.reset()

    def _save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
