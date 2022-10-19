import os

import torch.nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from hydra.utils import get_original_cwd, to_absolute_path

from .dataset import SemanticDataset
from .learning import Trainer, create_model, plot_results


def train_model(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_ds = SemanticDataset(cfg.path.kitti, cfg.kitti, 'train', cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.path.kitti, cfg.kitti, 'valid', cfg.train.dataset_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=os.cpu_count() // 2)

    model = create_model(cfg)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.train.learning_rate)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    history = trainer.train(cfg.train.n_epochs)

    plot_results(history, save_path='.')
