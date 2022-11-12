import logging

import numpy as np
from torchmetrics import MetricCollection
from torch.utils.tensorboard import SummaryWriter


class State:
    def __init__(self, writer: SummaryWriter, metrics: MetricCollection,
                 logger: logging.Logger, main_metric: str = 'JaccardIndex') -> None:

        # Logging
        self.writer = writer
        self.logger = logger

        # Loss
        self.loss = 0
        self.loss_history = []

        # Metrics
        self.metrics = None
        self.num_batches = 0
        self.main_metric = main_metric
        self.metric_collection = metrics
        self.metric_history = {metric_name: [] for metric_name in metrics.keys()}

    def accumulate_loss(self, loss) -> None:
        """ Update loss and keep track of number of batches """
        self.loss += loss
        self.num_batches += 1

    def accumulate_metrics(self, output, label_batch) -> None:
        """ Update torchmetrics """
        self.metric_collection(output, label_batch)

    def compute(self, phase: str) -> None:
        """ Compute loss and metrics and save them to history """

        # Compute loss and metrics
        if self.num_batches != 0:
            self.loss /= self.num_batches
        self.metrics = self.metric_collection.compute()

        # Save loss and metrics to history
        if phase != 'train':
            self.loss_history.append(self.loss)
            for metric_name, metric_value in self.metrics.items():
                self.metric_history[metric_name].append(metric_value.cpu().item())

    def reset(self) -> None:
        """ Reset loss and metrics """
        self.loss = 0
        self.metrics = None
        self.num_batches = 0
        self.metric_collection.reset()

    def stagnant(self, patience: int) -> bool:
        """ Check if loss has not decreased for a number of epochs """
        epochs_stagnated = len(self.loss_history) - np.argmin(self.loss_history) - 1
        self.logger.info(f'Epochs Stagnated: {epochs_stagnated}')
        return epochs_stagnated >= patience

    def main_metric_exceeded(self) -> bool:
        """ Check if main metric has improved """
        main_metric_history = self.metric_history[self.main_metric]
        return main_metric_history[-1] > max(main_metric_history)

    def log(self, index: int, phase: str) -> None:
        """ Log loss and metrics """

        # Log loss to tensorboard
        self.writer.add_scalar(f'Loss/{phase}', self.loss, index)

        # Log metrics to tensorboard
        for metric_name, metric_value in self.metrics.items():
            self.writer.add_scalar(f'{metric_name}/{phase}', metric_value, index)

        # Log loss and metrics to console
        if phase != 'test':
            out = f'\nEpoch {index} {phase} phase: \
                    \n\t- Loss: {self.loss}'
            for metric_name, metric_value in self.metrics.items():
                out += f'\n\t- {metric_name}: {metric_value}'
            print('')
            self.logger.info(out)
