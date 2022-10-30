import os
import logging

import torch.nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .dataset import SemanticDataset
from .learning import Trainer, create_model

log = logging.getLogger(__name__)


def train_model(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info(f'Using device {device}')

    train_ds = SemanticDataset(cfg.path.kitti, cfg.kitti, 'train', cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.path.kitti, cfg.kitti, 'valid', cfg.train.dataset_size)

    log.info(f'Train dataset size: {len(train_ds)}')
    log.info(f'Validation dataset size: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=os.cpu_count() // 2)

    model = create_model(cfg)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.train.learning_rate)

    log.info(f'Using criterion {criterion}')
    log.info(f'Using optimizer {optimizer}')

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, cfg.path.output)
    trainer.train(cfg.train.n_epochs, cfg.path.models)
