import os
import logging

import torch
import wandb
from omegaconf import DictConfig

from .trainer import SemanticTrainer
from src.datasets import SemanticDataset, PartitionDataset
from src.selection import get_selector
from src.utils import log_dataset_statistics

log = logging.getLogger(__name__)


def train_semantic_model(cfg: DictConfig, device: torch.device) -> None:
    """ Train model with fully labeled dataset.

    :param cfg: Config file
    :param device: Device to use for training
    """
    dataset_name = cfg.ds.name
    model_name = cfg.model.architecture
    # --------------------------------------------------------------------------------------------------
    # ========================================= Model Training =========================================
    # --------------------------------------------------------------------------------------------------
    with wandb.init(project=f'Train {model_name} {dataset_name}'):
        train_ds = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path,
                                   project_name='train_full', num_scans=cfg.train.dataset_size, al_experiment=False)
        val_ds = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path,
                                 project_name='val_full', num_scans=cfg.train.dataset_size, al_experiment=False)

        _, _, class_distribution, label_ratio = train_ds.statistics
        weights = 1 / (class_distribution + 1e-6)

        if abs(label_ratio - 1) > 1e-6:
            log.error(f'Label ratio is not 1: {label_ratio}')
            raise ValueError

        trainer = SemanticTrainer(cfg=cfg, train_ds=train_ds, val_ds=val_ds, device=device, weights=weights,
                                  model=None, model_name='semantic_model', history_name='semantic_history')
        trainer.train()


def train_partition_model(cfg: DictConfig, device: torch.device) -> None:
    raise NotImplementedError


def train_semantic_active(cfg: DictConfig, device: torch.device) -> None:
    """ Train model with active learning selection.

    :param cfg: Config file
    :param device: Device to use for training
    """

    # ================== Project Artifacts ==================
    model_artifact = cfg.active.model
    history_artifact = cfg.active.history
    selection_artifact = cfg.active.selection

    # ================== Project ==================
    criterion = cfg.active.criterion
    selection_objects = cfg.active.selection_objects
    project_name = f'{criterion}_{selection_objects}'
    percentage = f'{cfg.active.expected_percentage_labeled}%'
    load_model = cfg.load_model if 'load_model' in cfg else False

    # --------------------------------------------------------------------------------------------------
    # ========================================= Model Training =========================================
    # --------------------------------------------------------------------------------------------------

    with wandb.init(project=f'AL - {project_name}', group='training', name=f'Training - {percentage}'):

        # Load Datasets
        train_ds = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=project_name,
                                   num_scans=cfg.train.dataset_size, al_experiment=True, selection_mode=False)
        val_ds = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=project_name,
                                 num_scans=cfg.train.dataset_size, al_experiment=True, selection_mode=False)

        # Load Selector for selecting labeled voxels
        selector = get_selector(selection_objects=selection_objects, criterion=criterion,
                                dataset_path=cfg.ds.path, project_name=project_name,
                                cloud_paths=train_ds.clouds, device=device,
                                batch_size=cfg.active.batch_size)

        # Load selected voxels from W&B
        artifact_dir = wandb.use_artifact(f'{selection_artifact.name}:{selection_artifact.version}').download()
        selected_voxels = torch.load(os.path.join(artifact_dir, f'{selection_artifact.name}.pt'))

        # Label train dataset
        selector.load_voxel_selection(selected_voxels, train_ds)

        # Load model from W&B
        if load_model:
            artifact_dir = wandb.use_artifact(f'{model_artifact.name}:{model_artifact.version}').download()
            model = torch.load(os.path.join(artifact_dir, f'{model_artifact.name}.pt'), map_location=device)
        else:
            model = None

        # Log dataset statistics and calculate the weights for the loss function from them
        class_distribution = log_dataset_statistics(cfg=cfg, dataset=train_ds, save_artifact=True)
        weights = 1 / (class_distribution + 1e-6)

        # Train model
        trainer = SemanticTrainer(cfg=cfg, train_ds=train_ds, val_ds=val_ds, device=device, weights=weights,
                                  model=model, model_name=model_artifact.name, history_name=history_artifact.name)
        trainer.train()


def train_partition_active(cfg: DictConfig, device: torch.device) -> None:
    raise NotImplementedError
