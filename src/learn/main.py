import os
import logging

import torch
import wandb
from omegaconf import DictConfig

from src.datasets import SemanticDataset, SelectionDataset
from .trainer import Trainer
from src.models import get_model
from src.active_selectors import get_selector

log = logging.getLogger(__name__)


def train_semantic_model(cfg: DictConfig, device: torch.device):
    """ Train a semantic segmentation model.

    :param cfg: The configuration of the project.
    :param device: The device to train on.
    """

    resume = cfg.resume if 'resume' in cfg else False

    project_name = f'Semantic Model Training Dev'
    group_name = f'{cfg.model.architecture} {cfg.ds.name}'
    model_path = os.path.join(cfg.path.models, 'semantic', f'{cfg.model.architecture}_{cfg.ds.name}')

    run = wandb.init(project=project_name, group=group_name, resume=resume)

    train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, init=bool(~resume))
    val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size, init=bool(~resume))

    log.info(f'Loaded train dataset: \n{train_ds}')
    log.info(f'Loaded val dataset: \n{val_ds}')

    trainer = Trainer(cfg, train_ds, val_ds, device, model_path, resume)
    trainer.train()


# def train_semantic_model_active(cfg: DictConfig, device: torch.device):
#     """ Train a semantic segmentation model with active learning.
#
#     :param cfg: The configuration of the project.
#     :param device: The device to train on.
#     """
#     resume = cfg.resume if 'resume' in cfg else False
#
#     train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train',
#                                size=cfg.train.dataset_size, active_mode=True, resume=resume)
#     val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val',
#                              size=cfg.train.dataset_size, active_mode=True, resume=resume)
#     sel_ds = SelectionDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size)
#
#     log.info(f'Loaded train dataset: \n{train_ds}')
#     log.info(f'Loaded val dataset: \n{val_ds}')
#
#     method = 'entropy_voxels'
#     model_path = os.path.join(cfg.path.models, 'active_semantic',
#                               f'{cfg.model.architecture}_{cfg.ds.name}_{method}.pt')
#
#     trainer = ActiveTrainer(cfg, train_ds, val_ds, sel_ds, device, model_path, method, resume)
#     trainer.train()

def train_model(cfg: DictConfig, device: torch.device):
    load_model = False
    load_voxels = True
    state_artifact = 'state:latest'
    selected_voxels_artifact = 'selected_voxels:latest'

    with wandb.init(project='Viewpoint Entropy Active Learning', group='Training', name='Model training - 1%'):
        train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, active_mode=True)
        val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size, active_mode=True)
        selector = get_selector('entropy_voxels', train_ds.path, train_ds.get_dataset_clouds(), device)
        # with wandb.init(project='Main Active Learning', group='Fully Labeled Training', name='Base'):
        #     train_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, active_mode=False)
        #     val_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='val', size=cfg.train.dataset_size, active_mode=False)
        #     selector = get_selector('entropy_voxels', train_ds.path, train_ds.get_dataset_clouds(), device)

        if load_voxels:
            artifact = wandb.use_artifact(selected_voxels_artifact)
            artifact_dir = artifact.download()
            path = os.path.join(artifact_dir, f'selected_voxels.pt')
            selected_voxels = torch.load(path)
            selector.load_voxel_selection(selected_voxels, train_ds)

        if load_model:
            artifact = wandb.use_artifact(state_artifact)
            artifact_dir = artifact.download()
            path = os.path.join(artifact_dir, f'state.pt')
            state = torch.load(path, map_location=device)
        else:
            state = None

        trainer = Trainer(cfg, train_ds, val_ds, device, state)
        trainer.train()


def select_voxels(cfg: DictConfig, device: torch.device):
    percentage = '3%'
    select_percent = 1
    state_artifact = 'state:latest'
    selected_voxels_artifact = 'selected_voxels:latest'

    with wandb.init(project='Viewpoint Entropy Active Learning', group='Selection',
                    name=f'Entropy Voxel Selection - {percentage}'):
        dataset = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, active_mode=True)
        cloud_paths = dataset.get_dataset_clouds()
        selector = get_selector('entropy_voxels', dataset.path, cloud_paths, device)

        # Load the already selected voxels from W&B
        artifact = wandb.use_artifact(selected_voxels_artifact)
        artifact_dir = artifact.download()
        path = os.path.join(artifact_dir, f'selected_voxels.pt')
        selected_voxels = torch.load(path)

        # Set the already selected voxels
        selector.load_voxel_selection(selected_voxels)

        # Load the model from W&B
        artifact = wandb.use_artifact(state_artifact)
        artifact_dir = artifact.download()
        path = os.path.join(artifact_dir, f'state.pt')
        state = torch.load(path)
        model_state_dict = state['model_state_dict']
        model = get_model(cfg, device)
        model.load_state_dict(model_state_dict)

        # Select the next voxels
        selected_voxels = selector.select(dataset, model, select_percent)

        # Save the selected voxels to W&B
        torch.save(selected_voxels, 'selected_voxels.pt')
        artifact = wandb.Artifact('selected_voxels', type='dataset',
                                  description='The selected voxels for the next active learning iteration.')
        artifact.add_file('selected_voxels.pt')
        wandb.run.log_artifact(artifact)


def select_first_voxels(cfg: DictConfig, device: torch.device):
    with wandb.init(project='Viewpoint Entropy Active Learning', group='Selection', name='First 1% random selection'):
        # Create the selection dataset
        dataset = SemanticDataset(cfg.ds.path, cfg.ds, split='train', size=cfg.train.dataset_size, active_mode=True)

        # Create the selector
        cloud_paths = dataset.get_dataset_clouds()
        selector = get_selector('random_voxels', dataset.path, cloud_paths, device)

        selected_voxels = selector.select(dataset, 1)

        torch.save(selected_voxels, 'selected_voxels.pt')
        artifact = wandb.Artifact('selected_voxels', type='dataset',
                                  description='The selected voxels for the first active learning iteration.')
        artifact.add_file('selected_voxels.pt')
        wandb.run.log_artifact(artifact)
