import os
import logging

import torch.nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from .dataset import SemanticDataset
from .learning import Trainer, create_model, Selector

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

    acc = MulticlassAccuracy(num_classes=cfg.ds.num_classes, mdmc_average='samplewise').to(device)
    iou = MulticlassJaccardIndex(num_classes=cfg.ds.num_classes).to(device)
    metrics = MetricCollection([acc, iou])

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      metrics=metrics, device=device, log_path=cfg.path.output,
                      train_loader=train_loader, val_loader=val_loader)

    trainer.train(cfg.train.epochs, cfg.path.models)


def train_model_active(cfg: DictConfig):
    from copy import deepcopy

    def select_ids(loader, model, n_querry=4):
        selector = Selector(model=model, loader=loader, device=device)
        data = selector.calculate_entropies()

        ids = {int(i) for i in data[:n_querry, 1]}
        return ids

    model = create_model(cfg)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.train.learning_rate)

    acc = MulticlassAccuracy(num_classes=cfg.ds.num_classes, mdmc_average='samplewise').to(device)
    iou = MulticlassJaccardIndex(num_classes=cfg.ds.num_classes).to(device)
    metrics = MetricCollection([acc, iou])

    ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)

    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=cfg.train.dataset_size)

    unlabelled_ids = set(range(len(ds)))

    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=os.cpu_count() // 2)

    unlabelled_ds = deepcopy(ds)
    # TODO: need to be tested, change n_iters=5 based on active learning termination criteria
    for _ in range(5):
        unlabelled_loader = DataLoader(unlabelled_ds, batch_size=cfg.train.batch_size,
                                       shuffle=False, num_workers=os.cpu_count() // 2)

        log.info("Selecting data samples with high entropy...")
        train_ids = select_ids(loader=unlabelled_loader, model=model)
        unlabelled_ids = unlabelled_ids - train_ids

        # check if selected train indices do not intersect with the rest of indices
        assert train_ids.isdisjoint(unlabelled_ids)
        assert unlabelled_ids.isdisjoint(train_ids)

        train_ds = deepcopy(ds)
        train_ds.choose_data(list(train_ids))

        unlabelled_ds = deepcopy(ds)
        unlabelled_ds.choose_data(list(unlabelled_ids))

        log.info(f"Train dataset length: {len(train_ds)}")
        log.info(f"Test dataset length: {len(unlabelled_ds)}")

        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                                  shuffle=True, num_workers=os.cpu_count() // 2)

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

    acc = MulticlassAccuracy(num_classes=cfg.ds.num_classes, mdmc_average='samplewise').to(device)
    iou = MulticlassJaccardIndex(num_classes=cfg.ds.num_classes).to(device)
    metrics = MetricCollection([acc, iou])

    trainer = Trainer(model=model, metrics=metrics, device=device,
                      log_path=cfg.path.output, test_loader=test_loader)
    metric_history = trainer.test()

    for metric, values in metric_history.items():
        log.info(f'{metric}: {values}')
        log.info(f'{metric} mean: {sum(values) / len(values)}')
