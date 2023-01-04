import os
import logging
from copy import deepcopy

import torch.nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .model import SalsaNext
from .dataset import SemanticDataset
from .learning import Trainer, Selector

log = logging.getLogger(__name__)

gpu_count = torch.cuda.device_count()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log.info(f'Using device {device}')


def train_model(cfg: DictConfig):
    # Create datasets
    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=cfg.train.dataset_size)

    model = SalsaNext(cfg.ds.num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), cfg.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    trainer = Trainer(model=model, cfg=cfg, train_ds=train_ds,
                      val_ds=val_ds, optimizer=optimizer, scheduler=scheduler)

    trainer.train(cfg.train.epochs, cfg.path.models)


def train_model_active(cfg: DictConfig):
    print(cfg.active)

    ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=cfg.train.dataset_size)

    unlabelled_ids = set(range(len(ds)))

    model = SalsaNext(cfg.ds.num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), cfg.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    unlabelled_ds = deepcopy(ds)
    # TODO: need to be tested, change n_iters=5 based on active learning termination criteria
    for _ in range(5):
        unlabelled_loader = DataLoader(unlabelled_ds, batch_size=cfg.train.batch_size,
                                       shuffle=False, num_workers=gpu_count * 4)

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

        trainer = Trainer(model=model, cfg=cfg, train_ds=train_ds,
                          val_ds=val_ds, optimizer=optimizer, scheduler=scheduler)

        best_iou = trainer.train(cfg.train.epochs, cfg.path.models)
        print(f"Best IoU: {best_iou}")


def test_model(cfg):
    test_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid', size=cfg.test.dataset_size)

    model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
    model = SalsaNext(cfg.ds.num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    trainer = Trainer(model=model, cfg=cfg, val_ds=test_ds)
    trainer.test(vis=True)


def select_ids(loader, model, n_querry=1000):
    selector = Selector(model=model, loader=loader, device=device)
    entropies, indices = selector.calculate_entropies()

    ids = {int(i) for i in indices[:n_querry]}
    return ids


def check_termination_condition(cfg, best_iou, dataset_size):
    if cfg.active.termination_criterion == 'max_samples':
        return len(dataset_size) >= cfg.active.max_samples
    elif cfg.active.termination_criterion == 'min_iou':
        return best_iou >= cfg.active.min_iou
