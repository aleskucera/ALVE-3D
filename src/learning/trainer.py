import os
import logging

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, JaccardIndex, MetricCollection

from .state import State

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, output_path):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.train_loader = train_loader

        self.state = State(writer=SummaryWriter(output_path),
                           metrics=MetricCollection([Accuracy(mdmc_average='samplewise', top_k=1),
                                                     JaccardIndex(num_classes=20).to(device)]), logger=log)

    def train(self, epochs: int, save_path: str) -> None:
        for epoch in range(epochs):
            # ----------- Train Phase ------------
            self.model.train()
            for data in tqdm(self.train_loader):
                image_batch, label_batch = _parse_data(data, self.device)

                # Forward pass
                output = self.model(image_batch)['out']
                loss = self.criterion(output, label_batch)

                # Update metrics
                self.state.update_loss(loss.item())
                self.state.update_metrics(output, label_batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Average data and log to tensorboard and console
            self.state.compute()
            self.state.log(epoch, 'train', verbose=True)
            self.state.reset()

            # ----------- Validation Phase ------------
            self.model.eval()
            with torch.no_grad():
                for data in tqdm(self.val_loader):
                    image_batch, label_batch = _parse_data(data, self.device)

                    # Forward pass
                    output = self.model(image_batch)['out']

                    # Update metrics
                    loss = self.criterion(output, label_batch)
                    self.state.update_loss(loss.item())
                    self.state.update_metrics(output, label_batch)

            self.state.compute()
            self.state.save()
            self.state.log(epoch, 'val', verbose=True)
            self.state.reset()

            if self.state.main_metric_exceeded():
                log.info(f'Loss Decreasing, saving model to {save_path}')
                model_name = os.path.join(save_path, f'model_{epoch}.pth')
                self.save(os.path.join(save_path, model_name))

            if self.state.stagnant(patience=5):
                log.info('Training Stagnant, stopping training')
                break

            # Save model
            model_name = f'epoch-{epoch}.pth'
            self.save(os.path.join(save_path, model_name))

    def save(self, path):
        torch.save(self.model.state_dict(), path)


def _parse_data(data: tuple, device: torch.device):
    image_batch, label_batch = data
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)
    return image_batch, label_batch
