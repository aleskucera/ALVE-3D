import os
import logging

import torch
import wandb
from omegaconf import DictConfig

from .trainer import Trainer
from src.models import get_model
from src.datasets import SemanticDataset
from src.active_selectors import get_selector

log = logging.getLogger(__name__)


def train_model(cfg: DictConfig, device: torch.device):
    """ Train model with fully labeled dataset.

    :param cfg: Config file
    :param device: Device to use for training
    """

    with wandb.init(project='Train Semantic Model'):
        train_ds = SemanticDataset(cfg.ds.path, project_name='train_full', cfg=cfg.ds,
                                   split='train', size=cfg.train.dataset_size, active_mode=False)
        val_ds = SemanticDataset(cfg.ds.path, project_name='train_full', cfg=cfg.ds,
                                 split='val', size=cfg.train.dataset_size, active_mode=False)

        _, _, labeled_class_distribution, label_ratio = train_ds.get_statistics()
        weights = 1 / (labeled_class_distribution + 1e-6)

        if abs(label_ratio - 1) > 1e-6:
            log.error(f'Label ratio is not 1: {label_ratio}')
            raise ValueError

        trainer = Trainer(cfg, train_ds, val_ds, device, weights=weights, state=None)
        trainer.train()


def train_active(cfg: DictConfig, device: torch.device):
    """ Train model with active learning selection.

    :param cfg: Config file
    :param device: Device to use for training
    """

    # ================== Project Artifacts ==================
    model_artifact = cfg.active.model
    selected_voxels = cfg.active.selected_voxels
    load_model = cfg.load_model if 'load_model' in cfg else False

    # ================== Project ==================
    selector_type = cfg.active.selector_type
    project_name = f'AL - {selector_type}'
    percentage = f'{cfg.active.expected_percentage_labeled}%'

    # -----------------------------------------------------------------------------------------------
    # ========================================= Train Model =========================================
    # -----------------------------------------------------------------------------------------------

    with wandb.init(project=project_name, group='training', name=f'Training - {percentage}'):

        # Load Datasets
        train_ds = SemanticDataset(cfg.ds.path, project_name=selector_type, cfg=cfg.ds,
                                   split='train', size=cfg.train.dataset_size, active_mode=True)
        val_ds = SemanticDataset(cfg.ds.path, project_name=selector_type, cfg=cfg.ds,
                                 split='val', size=cfg.train.dataset_size, active_mode=True)

        # Load Selector for selecting labeled voxels
        selector = get_selector(selector_type, train_ds.path, train_ds.get_dataset_clouds(), device)

        # Load selected voxels from W&B
        artifact = wandb.use_artifact(f'{selected_voxels.name}:{selected_voxels.version}')
        artifact_dir = artifact.download()
        path = os.path.join(artifact_dir, f'{selected_voxels.name}.pt')
        selected_voxels = torch.load(path)

        # Label train dataset
        selector.load_voxel_selection(selected_voxels, train_ds)

        # Load model from W&B
        if load_model:
            artifact = wandb.use_artifact(f'{model_artifact.name}:{model_artifact.version}')
            artifact_dir = artifact.download()
            path = os.path.join(artifact_dir, f'{model_artifact.name}.pt')
            model = torch.load(path, map_location=device)
        else:
            model = None

        # Log dataset statistics and calculate the weights for the loss function from them
        labeled_class_distribution, label_ratio = selector.log_selection(cfg, train_ds, save=True)
        if abs(label_ratio * 100 - float(cfg.active.expected_percentage_labeled)) > 0.5:
            log.error(f'Label ratio is not {percentage}: {label_ratio * 100}')
            raise ValueError
        weights = 1 / (labeled_class_distribution + 1e-6)

        # Train model
        trainer = Trainer(cfg, train_ds, val_ds, device, weights, model)
        trainer.train()


def select_voxels(cfg: DictConfig, device: torch.device):
    """ Select voxels for active learning.

    :param cfg: Config file
    :param device: Device to use for training
    """

    # ================== Project Artifacts ==================
    model_artifact = cfg.active.model
    selected_voxels_artifact = cfg.active.selected_voxels
    load_model = cfg.load_model if 'load_model' in cfg else False

    # ================== Project ==================
    selector_type = cfg.active.selector_type
    project_name = f'AL - {selector_type}'
    select_percentage = cfg.active.select_percentage
    percentage = f'{cfg.active.expected_percentage_labeled + cfg.active.select_percentage}%'

    with wandb.init(project=project_name, group='selection', name=f'Selection - {percentage}'):
        dataset = SemanticDataset(cfg.ds.path, project_name=selector_type, cfg=cfg.ds,
                                  split='train', size=cfg.train.dataset_size, active_mode=True)
        selector = get_selector(selector_type, dataset.path, dataset.get_dataset_clouds(), device)

        # Load selected voxels from W&B
        artifact = wandb.use_artifact(f'{selected_voxels_artifact.name}:{selected_voxels_artifact.version}')
        artifact_dir = artifact.download()
        path = os.path.join(artifact_dir, f'{selected_voxels_artifact.name}.pt')
        selected_voxels = torch.load(path)

        # Label voxel clouds
        selector.load_voxel_selection(selected_voxels)

        # Load model from W&B
        if load_model:
            artifact = wandb.use_artifact(f'{model_artifact.name}:{model_artifact.version}')
            artifact_dir = artifact.download()
            path = os.path.join(artifact_dir, f'{model_artifact.name}.pt')
            model = torch.load(path, map_location=device)
            model_state_dict = model['model_state_dict']
            model = get_model(cfg, device)
            model.load_state_dict(model_state_dict)
        else:
            model = None

        # Select the next voxels
        selected_voxels = selector.select(dataset, model=model, percentage=select_percentage)

        # Save the selected voxels to W&B
        torch.save(selected_voxels, f'data/{selected_voxels_artifact.name}.pt')
        artifact = wandb.Artifact(selected_voxels_artifact.name, type='selection',
                                  description='The selected voxels for the next active learning iteration.')
        artifact.add_file(f'data/{selected_voxels_artifact.name}.pt')
        wandb.run.log_artifact(artifact)

        # Log the results of the selection to W&B
        selector.load_voxel_selection(selected_voxels, dataset)
        selector.log_selection(cfg, dataset)


def select_first_voxels(cfg: DictConfig, device: torch.device):
    """ Select the first voxels for active learning.

    :param cfg: Config file
    :param device: Device to use for training
    """

    # ================== Project ==================
    selector_type = cfg.active.selector_type
    random_selector_type = cfg.active.random_selector_type
    project_name = f'AL - {selector_type}'
    percentage = f'{cfg.active.expected_percentage_labeled + cfg.active.select_percentage}%'

    selected_voxels_artifact = cfg.active.selected_voxels
    with wandb.init(project=project_name, group='selection', name=f'First Selection - {percentage}'):
        dataset = SemanticDataset(cfg.ds.path, project_name=selector_type, cfg=cfg.ds,
                                  split='train', size=cfg.train.dataset_size, active_mode=True)
        cloud_paths = dataset.get_dataset_clouds()
        selector = get_selector(random_selector_type, dataset.path, cloud_paths, device)

        selected_voxels = selector.select(dataset, percentage=cfg.active.select_percentage)

        torch.save(selected_voxels, f'data/{selected_voxels_artifact.name}.pt')
        artifact = wandb.Artifact(selected_voxels_artifact.name, type='selection',
                                  description='The selected voxels for the first active learning iteration.')
        artifact.add_file(f'data/{selected_voxels_artifact.name}.pt')
        wandb.run.log_artifact(artifact)

        # Log the results of the first selection
        selector.load_voxel_selection(selected_voxels, dataset)
        selector.log_selection(cfg, dataset)
