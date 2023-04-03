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

        trainer = Trainer(cfg, train_ds, val_ds, device, state=None)
        trainer.train()


def train_active(cfg: DictConfig, device: torch.device):
    percentage = f'{cfg.active.percent_labeled}%'
    selector_type = cfg.active.selector_type
    project_name = f'active_learning_{selector_type}'

    load_model = cfg.load_model if 'load_model' in cfg else False
    load_voxels = cfg.load_voxels if 'load_voxels' in cfg else False
    state_artifact = cfg.active.state_artifact
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
            artifact = wandb.use_artifact(f'{state_artifact}:latest')
            artifact_dir = artifact.download()
            path = os.path.join(artifact_dir, f'{state_artifact}.pt')
            state = torch.load(path, map_location=device)
        else:
            state = None

        trainer = Trainer(cfg, train_ds, val_ds, device, state)
        trainer.train()


def select_voxels(cfg: DictConfig, device: torch.device):
    percentage = f'{cfg.active.percent_labeled}%'
    select_percent = cfg.active.select_percent
    selector_type = cfg.active.selector_type
    project_name = f'active_learning_{selector_type}'

    state_artifact = cfg.active.state_artifact
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
        artifact = wandb.use_artifact(f'{state_artifact}:latest')
        artifact_dir = artifact.download()
        path = os.path.join(artifact_dir, f'{state_artifact}.pt')
        state = torch.load(path)
        model_state_dict = state['model_state_dict']
        model = get_model(cfg, device)
        model.load_state_dict(model_state_dict)

        # Select the next voxels
        selected_voxels = selector.select(dataset, model, select_percent)

        # Save the selected voxels to W&B
        torch.save(selected_voxels, f'{selected_voxels_artifact}.pt')
        artifact = wandb.Artifact(selected_voxels_artifact, type='dataset',
                                  description='The selected voxels for the next active learning iteration.')
        artifact.add_file(f'{selected_voxels_artifact}.pt')
        wandb.run.log_artifact(artifact)


def select_first_voxels(cfg: DictConfig, device: torch.device):
    selector_type = cfg.active.selector_type
    random_selector_type = cfg.active.random_selector_type
    project_name = f'active_learning_{selector_type}'

    selected_voxels_artifact = cfg.active.selected_voxels_artifact
    with wandb.init(project=project_name, group='selection', name='first_random_selection_1%'):
        dataset = SemanticDataset(cfg.ds.path, project_name=project_name, cfg=cfg.ds,
                                  split='train', size=cfg.train.dataset_size, active_mode=True)
        cloud_paths = dataset.get_dataset_clouds()
        selector = get_selector(random_selector_type, dataset.path, cloud_paths, device)

        selected_voxels = selector.select(dataset, cfg.select_percent)

        torch.save(selected_voxels, f'{selected_voxels_artifact}.pt')
        artifact = wandb.Artifact(selected_voxels_artifact, type='dataset',
                                  description='The selected voxels for the first active learning iteration.')
        artifact.add_file(f'{selected_voxels_artifact}.pt')
        wandb.run.log_artifact(artifact)
