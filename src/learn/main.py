import os
import logging

import torch
import wandb
from omegaconf import DictConfig

from .trainer import Trainer
from src.models import get_model
from src.laserscan import LaserScan
from src.datasets import SemanticDataset
from src.active_selectors import get_selector
from src.utils import log_dataset_statistics, log_most_labeled_sample, log_selection_metric_statistics, log_selection

log = logging.getLogger(__name__)


def train_model(cfg: DictConfig, device: torch.device) -> None:
    """ Train model with fully labeled dataset.

    :param cfg: Config file
    :param device: Device to use for training
    """

    # ================== Project Artifacts ==================
    model_artifact = cfg.active.model
    history_artifact = cfg.active.history

    # --------------------------------------------------------------------------------------------------
    # ========================================= Model Training =========================================
    # --------------------------------------------------------------------------------------------------

    with wandb.init(project='Train Model'):
        train_ds = SemanticDataset(cfg.ds.path, project_name='train_full', cfg=cfg.ds,
                                   split='train', size=cfg.train.dataset_size, active_mode=False)
        val_ds = SemanticDataset(cfg.ds.path, project_name='train_full', cfg=cfg.ds,
                                 split='val', size=cfg.train.dataset_size, active_mode=False)

        _, _, class_distribution, label_ratio = train_ds.get_statistics()
        weights = 1 / (class_distribution + 1e-6)

        if abs(label_ratio - 1) > 1e-6:
            log.error(f'Label ratio is not 1: {label_ratio}')
            raise ValueError

        trainer = Trainer(cfg=cfg, train_ds=train_ds, val_ds=val_ds, device=device, weights=weights,
                          model=None, model_name=model_artifact.name, history_name=history_artifact.name)
        trainer.train()


def train_active(cfg: DictConfig, device: torch.device) -> None:
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

    with wandb.init(project=f'AL (test) - {project_name}', group='training', name=f'Training - {percentage}'):

        # Load Datasets
        train_ds = SemanticDataset(dataset_path=cfg.ds.path, project_name=project_name, cfg=cfg.ds, split='train',
                                   size=cfg.train.dataset_size, al_experiment=True, selection_mode=False)
        val_ds = SemanticDataset(dataset_path=cfg.ds.path, project_name=project_name, cfg=cfg.ds, split='val',
                                 size=cfg.train.dataset_size, al_experiment=True, selection_mode=False)

        # Load Selector for selecting labeled voxels
        selector = get_selector(selection_objects=selection_objects, criterion=criterion, dataset_path=cfg.ds.path,
                                cloud_paths=train_ds.voxel_clouds, device=device, batch_size=cfg.train.batch_size)

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
        trainer = Trainer(cfg=cfg, train_ds=train_ds, val_ds=val_ds, device=device, weights=weights,
                          model=model, model_name=model_artifact.name, history_name=history_artifact.name)
        trainer.train()


def select_voxels(cfg: DictConfig, device: torch.device) -> None:
    """ Select voxels for active learning.

    :param cfg: Config file
    :param device: Device to use for training
    """

    # ================== Project Artifacts ==================
    model_artifact = cfg.active.model
    selection_artifact = cfg.active.selection
    metric_statistics_artifact = cfg.active.metric_statistics

    # ================== Project ==================
    criterion = cfg.active.criterion
    selection_objects = cfg.active.selection_objects
    project_name = f'{criterion}_{selection_objects}'
    select_percentage = cfg.active.select_percentage
    percentage = f'{cfg.active.expected_percentage_labeled + cfg.active.select_percentage}%'

    with wandb.init(project=f'AL (test) - {project_name}', group='selection', name=f'Selection - {percentage}'):
        dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name=project_name,
                                  cfg=cfg.ds, split='train', size=cfg.train.dataset_size, al_experiment=True,
                                  selection_mode=True)
        selector = get_selector(selection_objects=selection_objects, criterion=criterion, dataset_path=dataset.path,
                                cloud_paths=dataset.voxel_clouds, device=device, batch_size=cfg.active.batch_size)

        # Load selected voxels from W&B
        artifact_dir = wandb.use_artifact(f'{selection_artifact.name}:{selection_artifact.version}').download()
        selection = torch.load(os.path.join(artifact_dir, f'{selection_artifact.name}.pt'))

        # Label voxel clouds
        selector.load_voxel_selection(selection)

        # Load model from W&B
        artifact_dir = wandb.use_artifact(f'{model_artifact.name}:{model_artifact.version}').download()
        model = torch.load(os.path.join(artifact_dir, f'{model_artifact.name}.pt'), map_location=device)
        model_state_dict = model['model_state_dict']
        model = get_model(cfg=cfg, device=device)
        model.load_state_dict(model_state_dict)

        # Select the next voxels
        selection, metric_statistics = selector.select(dataset=dataset, model=model, percentage=select_percentage)

        # Save the selection to W&B
        log_selection(selection=selection, selection_name=selection_artifact.name)

        # Save the statistics of the metric used for the selection to W&B
        if metric_statistics is not None:
            log_selection_metric_statistics(metric_statistics=metric_statistics,
                                            metric_statistics_name=metric_statistics_artifact.name)

        # Log the results of the selection to W&B
        selector.load_voxel_selection(voxel_selection=selection, dataset=dataset)
        log_dataset_statistics(cfg=cfg, dataset=dataset, save_artifact=False)

        scan = LaserScan(label_map=cfg.ds.learning_map,
                         color_map=cfg.ds.color_map_train,
                         colorize=True)
        log_most_labeled_sample(dataset=dataset, laser_scan=scan)


def select_first_voxels(cfg: DictConfig, device: torch.device) -> None:
    """ Select the first voxels for active learning.

    :param cfg: Config file
    :param device: Device to use for training
    """

    # ================== Project Artifacts ==================
    selection_artifact = cfg.active.selection

    # ================== Project ==================
    criterion = cfg.active.criterion
    selection_objects = cfg.active.selection_objects
    project_name = f'{criterion}_{selection_objects}'
    select_percentage = cfg.active.select_percentage
    percentage = f'{cfg.active.expected_percentage_labeled + cfg.active.select_percentage}%'

    with wandb.init(project=f'AL (test) - {project_name}', group='selection', name=f'First Selection - {percentage}'):
        dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name=project_name,
                                  cfg=cfg.ds, split='train', size=cfg.train.dataset_size, al_experiment=True,
                                  selection_mode=True)
        selector = get_selector(selection_objects=selection_objects, criterion='random', dataset_path=dataset.path,
                                cloud_paths=dataset.voxel_clouds, device=device, batch_size=cfg.active.batch_size)

        selection, _ = selector.select(dataset=dataset, percentage=select_percentage)

        # Save the selection to W&B
        log_selection(selection=selection, selection_name=selection_artifact.name)

        # Log the results of the first selection
        selector.load_voxel_selection(voxel_selection=selection, dataset=dataset)
        log_dataset_statistics(cfg=cfg, dataset=dataset, save_artifact=False)

        scan = LaserScan(label_map=cfg.ds.learning_map,
                         color_map=cfg.ds.color_map_train,
                         colorize=True)
        log_most_labeled_sample(dataset=dataset, laser_scan=scan)
