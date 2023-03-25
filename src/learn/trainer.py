import os
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
from src.active_selectors import get_selector

log = logging.getLogger(__name__)


class BaseTrainer(object):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset,
                 device: torch.device, model_path: str, resume: bool = False):
        self.cfg = cfg
        self.device = device
        self.val_ds = val_ds
        self.train_ds = train_ds
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)

        self.batch_size = cfg.train.batch_size
        self.num_workers = torch.cuda.device_count() * 4 if device.type == 'cuda' else 4

        self.model = get_model(cfg, device)
        self.loss_fn = get_loss(cfg.model.type, device)
        self.parser = get_parser(cfg.model.type, device)
        self.logger = get_logger(cfg.model.type, cfg.ds.num_classes, cfg.ds.labels_train, device, cfg.ds.ignore_index)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.learning_rate)

        self.epoch = 0

        if resume:
            self.load_state()

    def train(self):
        raise NotImplementedError

    def train_epoch(self, validate: bool = True) -> dict:
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
        return self.validate() if validate else {}

    def validate(self) -> dict:
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
        return self.logger.log_val(self.epoch)

    def save_state(self, path: str, history: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log.info(f'Saving model state to {path}...')

        # Create state dictionary and save it
        self.model.eval()
        state_dict = {'epoch': self.epoch,
                      'history': history,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(state_dict, path)

        # Create metadata for W&B artifact
        metadata = {'epoch': self.epoch}
        for key, value in history.items():
            metadata[key] = value

        # Create W&B artifact and log it
        artifact = wandb.Artifact(self.model_name, type='model', metadata=metadata,
                                  description='Model state with optimizer state and metric history for each epoch.')
        artifact.add_file(path)
        wandb.run.log_artifact(artifact)

    def load_state(self, path: str):
        if os.path.exists(path):
            state_path = path
        else:
            model_name = os.path.basename(path)
            artifact = wandb.use_artifact(f'{model_name}:latest')
            artifact_dir = artifact.download()
            state_path = os.path.join(artifact_dir, f'{model_name}.pt')

        log.info(f'Loading model state from {state_path}...')

        state = torch.load(state_path, map_location=self.device)
        self.epoch = state['epoch']
        self.logger.load_history(state['history'])
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])


class Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, device: torch.device,
                 model_path: str, resume: bool = False):
        super().__init__(cfg, train_ds, val_ds, device, model_path, resume)

    def train(self):
        while not self.logger.miou_converged:
            history = self.train_epoch(validate=True)
            if self.logger.miou_improved:
                self.save_state(self.model_path, history)
            self.epoch += 1


class ActiveTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, train_ds: Dataset, val_ds: Dataset, sel_ds: Dataset,
                 device: torch.device, model_path: str, method: str, resume: bool = False):
        super().__init__(cfg, train_ds, val_ds, device, model_path, resume)

        # Initialize the selector
        self.sel_ds = sel_ds
        self.method = method
        cloud_paths = train_ds.get_dataset_clouds()
        self.selector = get_selector(method, train_ds.path, cloud_paths, device)

        self.project_name = f'Active Semantic Model Training'
        self.group_name = f'Test logging'
        self.model_path = model_path

        # Save the initial state of the model and optimizer for train() to use
        # if not resume:
        #     self.save_state(self.model_path, self.logger.history)

    def train(self):
        end = False
        counter = 0
        last_labeled_ratio = -1
        while not end:
            # Load the model state and start the wandb run reset logger
            # self.load_state(self.model_path)
            self.logger.reset()

            # Select the next labels to be labeled with loaded model
            with wandb.init(project='select'):
                self.selector.select(self.train_ds, self.sel_ds, self.model)

            # Log the new dataset statistics
            class_distribution, class_progress, labeled_ratio = self.train_ds.get_statistics()
            if labeled_ratio == last_labeled_ratio or labeled_ratio >= 0.95:
                end = True
            last_labeled_ratio = labeled_ratio
            print(f'Labeled ratio: {labeled_ratio:.2f}')

            with wandb.init(project=self.project_name, group=self.group_name, name=f'{self.method}_{counter}'):
                self.logger.log_dataset_statistics(class_distribution, class_progress, labeled_ratio)

            # Train the model until convergence
            while not self.logger.miou_converged:
                history = self.train_epoch(validate=True)
                if self.logger.miou_improved:
                    self.save_state(self.model_path, history)
                self.epoch += 1
            counter += 1
