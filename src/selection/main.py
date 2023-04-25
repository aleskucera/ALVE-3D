import os

import torch
import wandb
import numpy as np
from omegaconf import DictConfig

from .base_selector import Selector
from .voxel_selector import VoxelSelector
from .superpoint_selector import SuperpointSelector

from src.models import get_model
from src.laserscan import LaserScan
from src.datasets import SemanticDataset
from src.utils.experiment import Experiment
from src.utils.log import log_dataset_statistics, log_most_labeled_sample, \
    log_selection_metric_statistics, log_selection


def get_selector(selection_objects: str, criterion: str, dataset_path: str, project_name: str,
                 cloud_paths: np.ndarray, device: torch.device, batch_size: int) -> Selector:
    if selection_objects == 'Voxels':
        return VoxelSelector(dataset_path, project_name, cloud_paths, device, criterion, batch_size)
    elif selection_objects == 'Superpoints':
        return SuperpointSelector(dataset_path, project_name, cloud_paths, device, criterion, batch_size)
    else:
        raise ValueError(f'Unknown selection_objects: {selection_objects}')


def select_voxels(cfg: DictConfig, experiment: Experiment, device: torch.device) -> None:
    """ Selects the voxels to be labeled for the next training iteration.
    The function executes the following steps:
        1. Load the dataset.
        2. Create a Selector object. If the expected percentage of labeled voxels is equal to 0, the selector will be a
           RandomSelector. Otherwise, it will be a selector based on the criterion specified in the configuration file.
        3. If the expected percentage of labeled voxels is greater than 0, load the voxel selection from W&B
           and use it to tell the selector which voxels are already labeled. Then, load the model from W&B
           and use it to select the voxels to be labeled.
        4. Upload the voxel selection to W&B with the statistics of the selection.
        5. Upload the most labeled sample to W&B.

    :param cfg: The configuration object containing the dataset parameters.
    :param experiment: The experiment object containing the names of the artifacts to be used.
    :param device: The device to be used for the selection.
    """

    criterion = cfg.active.criterion
    selection_objects = cfg.active.selection_objects
    select_percentage = cfg.active.select_percentage
    expected_percentage_labeled = cfg.active.expected_percentage_labeled

    model_version = cfg.active.model_version
    selection_version = cfg.active.selection_version

    dataset = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=experiment.info,
                              num_clouds=cfg.train.dataset_size, al_experiment=True, selection_mode=True)

    if expected_percentage_labeled > 0:
        selector = get_selector(selection_objects=selection_objects, criterion=criterion,
                                dataset_path=cfg.ds.path, project_name=experiment.info,
                                cloud_paths=dataset.cloud_files, device=device, batch_size=cfg.active.batch_size)

        artifact_dir = wandb.use_artifact(f'{experiment.selection}:{selection_version}').download()
        selection = torch.load(os.path.join(artifact_dir, f'{experiment.selection}.pt'))

        selector.load_voxel_selection(selection, dataset)

        # Load model from W&B
        artifact_dir = wandb.use_artifact(f'{experiment.model}:{model_version}').download()
        model = torch.load(os.path.join(artifact_dir, f'{experiment.model}.pt'), map_location=device)
        model_state_dict = model['model_state_dict']
        model = get_model(cfg=cfg, device=device)
        model.load_state_dict(model_state_dict)

        selection, metric_statistics = selector.select(dataset=dataset, model=model, percentage=select_percentage)

    else:
        selector = get_selector(selection_objects=selection_objects, criterion='Random',
                                dataset_path=cfg.ds.path, project_name=experiment.info,
                                cloud_paths=dataset.cloud_files, device=device, batch_size=cfg.active.batch_size)

        selection, metric_statistics = selector.select(dataset=dataset, percentage=select_percentage)

    # Save the selection to W&B
    log_selection(selection=selection, selection_name=experiment.selection)

    # Save the statistics of the metric used for the selection to W&B
    if metric_statistics is not None:
        log_selection_metric_statistics(metric_statistics=metric_statistics,
                                        metric_statistics_name=experiment.metric_stats)

    # Log the results of the selection to W&B
    selector.load_voxel_selection(voxel_selection=selection, dataset=dataset)
    log_dataset_statistics(cfg=cfg, dataset=dataset)

    scan = LaserScan(label_map=cfg.ds.learning_map,
                     color_map=cfg.ds.color_map_train,
                     colorize=True)
    log_most_labeled_sample(dataset=dataset, laser_scan=scan)
