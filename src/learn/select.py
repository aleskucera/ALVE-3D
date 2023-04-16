import os

import torch
import wandb
from omegaconf import DictConfig

from src.models import get_model
from src.laserscan import LaserScan
from src.datasets import SemanticDataset
from src.selection import get_selector
from src.utils import log_dataset_statistics, log_most_labeled_sample, log_selection_metric_statistics, log_selection


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

    with wandb.init(project=f'AL - {project_name}', group='selection', name=f'Selection - {percentage}'):
        dataset = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=project_name,
                                  num_scans=cfg.train.dataset_size, al_experiment=True, selection_mode=True)
        selector = get_selector(selection_objects=selection_objects, criterion=criterion,
                                dataset_path=cfg.ds.path, project_name=project_name,
                                cloud_paths=dataset.clouds, device=device,
                                batch_size=cfg.active.batch_size)

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

    with wandb.init(project=f'AL - {project_name}', group='selection', name=f'First Selection - {percentage}'):
        dataset = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=project_name,
                                  num_scans=cfg.train.dataset_size, al_experiment=True, selection_mode=False)
        selector = get_selector(selection_objects=selection_objects, criterion='random',
                                dataset_path=cfg.ds.path, project_name=project_name,
                                cloud_paths=dataset.clouds, device=device,
                                batch_size=cfg.active.batch_size)

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
