import os
import logging

import torch.nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, JaccardIndex, MetricCollection

from .dataset import SemanticDataset
from .learning import Trainer, create_model

log = logging.getLogger(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log.info(f'Using device {device}')


def train_model(cfg: DictConfig):
    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=cfg.train.dataset_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=os.cpu_count() // 2)

    model = create_model(cfg)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.train.learning_rate)
    metrics = MetricCollection([Accuracy(mdmc_average='samplewise', top_k=1),
                                JaccardIndex(num_classes=cfg.ds.num_classes).to(device)])

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      metrics=metrics, device=device, log_path=cfg.path.output,
                      train_loader=train_loader, val_loader=val_loader)

    trainer.train(cfg.train.epochs, cfg.path.models)


def test_model(cfg):
    test_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=cfg.test.dataset_size)

    test_loader = DataLoader(test_ds, batch_size=cfg.test.batch_size,
                             shuffle=False, num_workers=os.cpu_count() // 2)

    model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
    model = torch.load(model_path).to(device)

    metrics = MetricCollection([Accuracy(mdmc_average='samplewise', top_k=1),
                                JaccardIndex(num_classes=cfg.ds.num_classes).to(device)])

    trainer = Trainer(model=model, metrics=metrics, device=device,
                      log_path=cfg.path.output, test_loader=test_loader)
    metric_history = trainer.test()

    for metric, values in metric_history.items():
        log.info(f'{metric}: {values}')
        log.info(f'{metric} mean: {sum(values) / len(values)}')
