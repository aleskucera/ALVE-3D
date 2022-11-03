import logging

import numpy as np
from torchmetrics import MetricCollection
from torch.utils.tensorboard import SummaryWriter


class State:
    def __init__(self, writer: SummaryWriter, metrics: MetricCollection, logger: logging.Logger,
                 main_metric: str = 'JaccardIndex') -> None:

        self.writer = writer
        self.logger = logger
        self.metric_collection = metrics

        self.loss = 0
        self.metrics = None
        self.num_batches = 0

        self.losses = []
        self.main_metrics = []
        self.main_metric = main_metric

    def update_loss(self, loss) -> None:
        self.loss += loss
        self.num_batches += 1

    def update_metrics(self, output, label_batch) -> None:
        self.metric_collection(output, label_batch)

    def log(self, epoch: int, phase: str, verbose: bool = False) -> None:
        self.writer.add_scalar(f'Loss/{phase}', self.loss, epoch)
        for metric_name, metric_value in self.metrics.items():
            self.writer.add_scalar(f'{metric_name}/{phase}', metric_value, epoch)

        if verbose:
            out = f'\nEpoch {epoch} {phase} phase: \
                    \n\t- Loss: {self.loss}'
            for metric_name, metric_value in self.metrics.items():
                out += f'\n\t- {metric_name}: {metric_value}'
            self.logger.info(out)

    def compute(self, ) -> None:
        self.loss /= self.num_batches
        self.metrics = self.metric_collection.compute()

    def save(self):
        self.losses.append(self.loss)
        self.main_metrics.append(self.metrics[self.main_metric])

    def reset(self) -> None:
        self.loss = 0
        self.metrics = None
        self.num_batches = 0
        self.metric_collection.reset()

    def stagnant(self, patience: int) -> bool:
        epochs_stagnated = len(self.losses) - np.argmin(self.losses) - 1
        self.logger.info(f'Epochs Stagnated: {epochs_stagnated}')
        return epochs_stagnated >= patience

    def min_loss_exceeded(self) -> bool:
        return self.losses[-1] < min(self.losses)

    def main_metric_exceeded(self) -> bool:
        return self.main_metrics[-1] > max(self.main_metrics)
