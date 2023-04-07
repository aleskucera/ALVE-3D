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
    percentage = f'{cfg.active.expected_percentage_labeled}%'
    selector_type = cfg.active.selector_type
    project_name = f'active_learning_{selector_type}'

    load_model = cfg.load_model if 'load_model' in cfg else False
    load_voxels = cfg.load_voxels if 'load_voxels' in cfg else False
    model_artifact = cfg.active.model_artifact
    selected_voxels_artifact = cfg.active.selected_voxels_artifact

    with wandb.init(project=project_name, group='training', name=f'model_training_{percentage}'):
        train_ds = SemanticDataset(cfg.ds.path, project_name=project_name, cfg=cfg.ds,
                                   split='train', size=cfg.train.dataset_size, active_mode=True)
        val_ds = SemanticDataset(cfg.ds.path, project_name=project_name, cfg=cfg.ds,
                                 split='val', size=cfg.train.dataset_size, active_mode=True)
        selector = get_selector(selector_type, train_ds.path, train_ds.get_dataset_clouds(), device)

        if load_voxels:
            artifact = wandb.use_artifact(f'{selected_voxels_artifact}:latest')
            artifact_dir = artifact.download()
            path = os.path.join(artifact_dir, f'{selected_voxels_artifact}.pt')
            selected_voxels = torch.load(path)
            selector.load_voxel_selection(selected_voxels, train_ds)

        if load_model:
            artifact = wandb.use_artifact(f'{model_artifact}:latest')
            artifact_dir = artifact.download()
            path = os.path.join(artifact_dir, f'{model_artifact}.pt')
            state = torch.load(path, map_location=device)
        else:
            state = None

        labeled_class_distribution, label_ratio = selector.log_selection(cfg, train_ds, save=True)
        if abs(label_ratio * 100 - float(cfg.active.expected_percentage_labeled)) > 0.5:
            log.error(f'Label ratio is not {percentage}: {label_ratio * 100}')
            raise ValueError
        weights = 1 / (labeled_class_distribution + 1e-6)

        trainer = Trainer(cfg, train_ds, val_ds, device, weights, state)
        trainer.train()


def select_voxels(cfg: DictConfig, device: torch.device):
    percentage = f'{cfg.active.expected_percentage_labeled + cfg.active.select_percentage}%'
    select_percentage = cfg.active.select_percentage
    selector_type = cfg.active.selector_type
    project_name = f'active_learning_{selector_type}'

    model_artifact = cfg.active.model_artifact
    selected_voxels_artifact = cfg.active.selected_voxels_artifact

    with wandb.init(project=project_name, group='selection', name=f'selection_{percentage}'):
        dataset = SemanticDataset(cfg.ds.path, project_name=project_name, cfg=cfg.ds,
                                  split='train', size=cfg.train.dataset_size, active_mode=True)
        cloud_paths = dataset.get_dataset_clouds()
        selector = get_selector(selector_type, dataset.path, cloud_paths, device)

        # Load the already selected voxels from W&B
        artifact = wandb.use_artifact(f'{selected_voxels_artifact}:latest')
        artifact_dir = artifact.download()
        path = os.path.join(artifact_dir, f'{selected_voxels_artifact}.pt')
        selected_voxels = torch.load(path)

        # Set the already selected voxels
        selector.load_voxel_selection(selected_voxels)

        # Load the model from W&B
        artifact = wandb.use_artifact(f'{model_artifact}:latest')
        artifact_dir = artifact.download()
        path = os.path.join(artifact_dir, f'{model_artifact}.pt')
        state = torch.load(path)
        model_state_dict = state['model_state_dict']
        model = get_model(cfg, device)
        model.load_state_dict(model_state_dict)

        # Select the next voxels
        selected_voxels = selector.select(dataset, model, select_percentage)

        # Save the selected voxels to W&B
        torch.save(selected_voxels, f'data/{selected_voxels_artifact}.pt')
        artifact = wandb.Artifact(selected_voxels_artifact, type='dataset',
                                  description='The selected voxels for the next active learning iteration.')
        artifact.add_file(f'data/{selected_voxels_artifact}.pt')
        wandb.run.log_artifact(artifact)

        # Log the results of the selection
        selector.load_voxel_selection(selected_voxels, dataset)
        selector.log_selection(cfg, dataset)


def select_first_voxels(cfg: DictConfig, device: torch.device):
    percentage = f'{cfg.active.expected_percentage_labeled + cfg.active.select_percentage}%'

    selector_type = cfg.active.selector_type
    random_selector_type = cfg.active.random_selector_type
    project_name = f'active_learning_{selector_type}'

    selected_voxels_artifact = cfg.active.selected_voxels_artifact
    with wandb.init(project=project_name, group='selection', name=f'first_random_selection_{percentage}'):
        dataset = SemanticDataset(cfg.ds.path, project_name=project_name, cfg=cfg.ds,
                                  split='train', size=cfg.train.dataset_size, active_mode=True)
        cloud_paths = dataset.get_dataset_clouds()
        selector = get_selector(random_selector_type, dataset.path, cloud_paths, device)

        selected_voxels = selector.select(dataset, cfg.active.select_percentage)

        torch.save(selected_voxels, f'data/{selected_voxels_artifact}.pt')
        artifact = wandb.Artifact(selected_voxels_artifact, type='dataset',
                                  description='The selected voxels for the first active learning iteration.')
        artifact.add_file(f'data/{selected_voxels_artifact}.pt')
        wandb.run.log_artifact(artifact)

        # Log the results of the first selection
        selector.load_voxel_selection(selected_voxels, dataset)
        selector.log_selection(cfg, dataset)
