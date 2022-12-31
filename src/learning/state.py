import logging

import torch
import numpy as np
from torchmetrics import MetricCollection
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from src.laserscan import ScanVis, LaserScan


class State:
    def __init__(self, writer: SummaryWriter, metrics: MetricCollection,
                 logger: logging.Logger, scan: LaserScan) -> None:

        # Logging
        self.writer = writer
        self.logger = logger

        # LaserScan
        self.scan = scan

        # MetricCollection
        self.metric_collection = metrics

        # Current state
        self.loss = 0
        self.metrics = {}
        self.num_batches = 0

        # History
        self.metric_history = {metric_name: [] for metric_name in metrics.keys()}

        # Worst samples
        self.min_iou = float('inf')
        self.worst_samples = None
        self.worst_predictions = None

    def accumulate_loss(self, loss) -> None:
        """ Update loss and keep track of number of batches """
        self.loss += loss
        self.num_batches += 1

    def accumulate_metrics(self, output, label_batch, indices) -> None:
        """ Update torchmetrics """
        metrics = self.metric_collection(output, label_batch)

        # Update worst samples
        if metrics['MulticlassJaccardIndex'] < self.min_iou:
            self.min_iou = metrics['MulticlassJaccardIndex']
            self.worst_predictions = output
            self.worst_samples = indices

    def compute(self, phase: str) -> None:
        """ Compute loss and metrics and save them to history """

        # Compute loss
        if self.num_batches != 0:
            self.loss /= self.num_batches

        # Compute metrics
        self.metrics = self.metric_collection.compute()

        # Save metrics to history
        if phase != 'train':
            for metric_name, metric_value in self.metrics.items():
                self.metric_history[metric_name].append(metric_value.cpu().item())

    def reset(self) -> None:
        """ Reset loss and metrics """
        self.loss = 0
        self.num_batches = 0

        self.metric_collection.reset()

        self.min_iou = float('inf')
        self.worst_samples = None
        self.worst_predictions = None

    def best_iou(self) -> bool:
        """ Check if main metric has improved """
        main_metric_history = self.metric_history['MulticlassJaccardIndex']
        if len(main_metric_history) > 1:
            return main_metric_history[-1] > max(main_metric_history[:-1])
        else:
            return False

    def get_best_iou(self) -> float:
        """ Return best iou """
        return max(self.metric_history['MulticlassJaccardIndex'])

    def log(self, dataset, epoch: int, phase: str, vis: False) -> None:
        """ Log loss and metrics """
        self._log_to_console(epoch, phase)
        self._log_to_tensorboard(epoch, phase)
        self._log_worst_predictions(dataset, epoch, phase, vis)

    def _log_to_tensorboard(self, epoch: int, phase: str) -> None:
        """ Log loss and metrics to tensorboard """

        # Log loss to tensorboard
        if self.loss != 0:
            self.writer.add_scalar(f'Loss/{phase}', self.loss, epoch)

        # Log metrics to tensorboard
        for metric_name, metric_value in self.metrics.items():
            self.writer.add_scalar(f'{metric_name}/{phase}', metric_value, epoch)

    def _log_to_console(self, epoch: int, phase: str) -> None:
        """ Log loss and metrics to console """

        # Intro
        if phase != 'test':
            out = f'\nEpoch {epoch} {phase} phase:'
        else:
            out = f'\n Result on test set:'

        # Loss
        if self.loss != 0:
            out += f'\n\t- Loss: {self.loss}'

        # Metrics
        for metric_name, metric_value in self.metrics.items():
            out += f'\n\t- {metric_name}: {metric_value}'

        # Print output to console
        print('')
        self.logger.info(out)

    def _log_worst_predictions(self, dataset, epoch: int, phase: str, vis: False) -> None:

        # Retrieve the worst predictions
        points = [dataset.points[idx] for idx in self.worst_samples]
        labels = [dataset.labels[idx] for idx in self.worst_samples]
        predictions = torch.argmax(self.worst_predictions, dim=1)

        points = points[:5]
        labels = labels[:5]
        predictions = predictions[:5]

        # Visualize the worst predictions and save to tensorboard
        for i, (point, label, prediction) in enumerate(zip(points, labels, predictions)):
            self.scan.open_points(point)
            self.scan.open_label(label)
            self.scan.set_prediction(prediction)

            # Visualize difference between prediction and label
            mask = np.all(self.scan.proj_sem_color != self.scan.proj_pred_color, axis=2, keepdims=True).astype(np.uint8)
            difference = self.scan.proj_sem_color * mask

            # Create figure and add subplots
            fig = plt.figure(figsize=(11, 4), dpi=150)
            grid = ImageGrid(fig, 111, nrows_ncols=(4, 1), axes_pad=0.4)

            images = [self.scan.proj_color, self.scan.proj_sem_color[..., ::-1],
                      self.scan.proj_pred_color[..., ::-1], difference[..., ::-1]]
            titles = ['Lidar', 'Ground Truth', 'Prediction', 'Difference']

            for ax, image, title in zip(grid, images, titles):
                ax.set_title(title)
                ax.imshow(image, aspect='auto')
                ax.axis('off')

            # Save figure to tensorboard
            self.writer.add_figure(f'Epoch: {epoch}/{phase}', fig, global_step=i)

        if vis:
            scan_vis = ScanVis(scan=self.scan, scans=points, labels=labels, predictions=predictions)
            scan_vis.run()
