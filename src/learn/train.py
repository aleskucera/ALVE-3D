import os
import logging

import torch
import wandb
from omegaconf import DictConfig

from .trainer import SemanticTrainer
from src.datasets import SemanticDataset
from src.selection import get_selector
from src.utils import log_dataset_statistics, Experiment

log = logging.getLogger(__name__)


def train_semantic_model(cfg: DictConfig, experiment: Experiment, device: torch.device) -> None:
    train_ds = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path,
                               project_name=experiment.info, num_clouds=cfg.train.dataset_size, al_experiment=False)
    val_ds = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path,
                             project_name=experiment.info, num_clouds=cfg.train.dataset_size, al_experiment=False)

    stats = train_ds.statistics
    weights = 1 / (stats['class_distribution'] + 1e-6)

    if abs(stats['labeled_ratio'] - 1) > 1e-6:
        log.error(f'Label ratio is not 1: {stats["labeled_ratio"]}')
        raise ValueError

    trainer = SemanticTrainer(cfg=cfg, train_ds=train_ds, val_ds=val_ds, device=device, weights=weights,
                              model=None, model_name=experiment.model, history_name=experiment.history)
    trainer.train()


def train_partition_model(cfg: DictConfig, experiment: Experiment, device: torch.device) -> None:
    raise NotImplementedError


def train_semantic_active(cfg: DictConfig, experiment: Experiment, device: torch.device) -> None:
    criterion = cfg.active.criterion
    selection_objects = cfg.active.selection_objects

    model_version = cfg.active.model_version
    selection_version = cfg.active.selection_version

    load_model = cfg.load_model if 'load_model' in cfg else False

    # Load Datasets
    train_ds = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=experiment.info,
                               num_clouds=cfg.train.dataset_size, al_experiment=True, selection_mode=False)
    val_ds = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=experiment.info,
                             num_clouds=cfg.train.dataset_size, al_experiment=True, selection_mode=False)

    # Load Selector for selecting labeled voxels
    selector = get_selector(selection_objects=selection_objects, criterion=criterion,
                            dataset_path=cfg.ds.path, project_name=experiment.info,
                            cloud_paths=train_ds.cloud_files, device=device,
                            batch_size=cfg.active.batch_size)

    # Load selected voxels from W&B
    artifact_dir = wandb.use_artifact(f'{experiment.selection}:{selection_version}').download()
    selected_voxels = torch.load(os.path.join(artifact_dir, f'{experiment.selection}.pt'))

    # Label train dataset
    selector.load_voxel_selection(selected_voxels, train_ds)

    print(train_ds)

    # Load model from W&B
    if load_model:
        artifact_dir = wandb.use_artifact(f'{experiment.model}:{model_version}').download()
        model = torch.load(os.path.join(artifact_dir, f'{experiment.model}.pt'), map_location=device)
    else:
        model = None

    # Log dataset statistics and calculate the weights for the loss function from them
    class_distribution = log_dataset_statistics(cfg=cfg, dataset=train_ds, artifact_name=experiment.dataset_stats)
    weights = 1 / (class_distribution + 1e-6)

    # Train model
    trainer = SemanticTrainer(cfg=cfg, train_ds=train_ds, val_ds=val_ds, device=device, weights=weights,
                              model=model, model_name=experiment.model, history_name=experiment.history)
    trainer.train()


def train_partition_active(cfg: DictConfig, experiment: Experiment, device: torch.device) -> None:
    raise NotImplementedError
