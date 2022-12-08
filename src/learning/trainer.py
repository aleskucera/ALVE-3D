import os
import logging

import torch
import matplotlib
from tqdm import tqdm

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torch.utils.tensorboard import SummaryWriter

from .state import State
from .utils import parse_data
from laserscan import LaserScan
from dataset import SemanticDataset

matplotlib.use('Qt5Agg')

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    device = torch.device('cuda')
else:
    gpu_count = 0
    device = torch.device('cpu')

log = logging.getLogger(__name__)


def train_requirements(func):
    def wrapper(self, *args, **kwargs):
        if self.train_ds is None:
            raise RuntimeError('Train dataset is not set')
        if self.criterion is None:
            raise RuntimeError('Criterion is not set')
        if self.optimizer is None:
            raise RuntimeError('Optimizer is not set')
        if self.scheduler is None:
            raise RuntimeError('Scheduler is not set')
        log.info('All requirements for training are met, starting training')
        return func(self, *args, **kwargs)

    return wrapper


class Trainer:
    def __init__(self, model: torch.nn.Module, metrics: MetricCollection,
                 cfg: DictConfig, val_ds: SemanticDataset, train_ds: SemanticDataset = None,
                 criterion: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None,
                 scheduler: object = None):

        self.cfg = cfg
        self.model = model
        self.val_ds = val_ds
        self.train_ds = train_ds

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = SummaryWriter(cfg.path.output)
        self.scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)
        self.state = State(writer=self.writer, metrics=metrics, logger=log, scan=self.scan)

    @train_requirements
    def train(self, epochs: int, save_path: str) -> None:
        """ Train model for a number of epochs and save the best models to path """

        # Create dataloaders
        train_loader = DataLoader(self.train_ds, batch_size=self.cfg.train.batch_size,
                                  shuffle=True, num_workers=gpu_count * 4)
        val_loader = DataLoader(self.val_ds, batch_size=self.cfg.train.batch_size,
                                shuffle=False, num_workers=gpu_count * 4)

        # Train for a number of epochs
        for epoch in range(epochs):

            # Train and validate
            self._train_epoch(train_loader, epoch)
            self._val_epoch(val_loader, epoch)

            # Save best models
            if self.state.best_iou():
                iou = self.state.metrics["MulticlassJaccardIndex"].item()
                model_name = f'{self.cfg.model.architecture}_e-{epoch}_iou-{iou:.3f}.pth'
                model_path = os.path.join(save_path, model_name)
                log.info(f'New best model found, saving to {model_path}')
                self._save_model(model_path)

    def test(self, vis: bool = False) -> None:
        """ Test model on test set """
        val_loader = DataLoader(self.val_ds, batch_size=self.cfg.test.batch_size,
                                shuffle=False, num_workers=gpu_count * 4)

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                image_batch, label_batch, indices = parse_data(data, device)

                # Forward pass (don't need loss for backward pass)
                output = self.model(image_batch)['out']
                self.state.accumulate_metrics(output, label_batch, indices)

        # Compute metrics, log to tensorboard, console and reset
        self._compute_and_log(0, 'test', vis=vis)

    def _train_epoch(self, loader, epoch: int) -> None:
        """ Train model on training set """
        self.model.train()
        for data in tqdm(loader):
            # Extract only inputs and labels
            image_batch, label_batch, indices = parse_data(data, device)

            # Forward pass
            loss, output = self._forward_pass(image_batch, label_batch, indices)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self._compute_and_log(epoch, 'train')

    def _val_epoch(self, loader, epoch: int) -> None:
        """ Validate model on validation set """
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(loader):
                # Extract only inputs and labels
                image_batch, label_batch, indices = parse_data(data, device)

                # Forward pass (don't need loss for backward pass)
                _, _ = self._forward_pass(image_batch, label_batch, indices)

        self._compute_and_log(epoch, 'val')

    def _forward_pass(self, image_batch: torch.Tensor, label_batch: torch.Tensor,
                      indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Forward pass
        output = self.model(image_batch)['out']
        loss = self.criterion(output, label_batch)

        # Accumulate loss and metrics
        self.state.accumulate_loss(loss.item())
        self.state.accumulate_metrics(output, label_batch, indices)
        return loss, output

    def _compute_and_log(self, epoch: int, phase: str, vis=False) -> None:
        self.state.compute(phase)
        if phase == 'train':
            self.state.log(self.train_ds, epoch, phase, vis=vis)
        else:
            self.state.log(self.val_ds, epoch, phase, vis=vis)
        self.state.reset()

    def _save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
