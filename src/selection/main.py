import os
import logging

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

log = logging.getLogger(__name__)


def get_selector(cfg: DictConfig, project_name: str, cloud_paths: np.ndarray, device: torch.device, ) -> Selector:
    if cfg.active.cloud_partitions == 'Voxels':
        return VoxelSelector(cfg, project_name, cloud_paths, device)
    elif cfg.active.cloud_partitions == 'Superpoints':
        return SuperpointSelector(cfg, project_name, cloud_paths, device)
    else:
        raise ValueError(f'Unknown selection_objects: {cfg.active.cloud_partitions}')


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

    seed = cfg.active.seed
    criterion = cfg.active.strategy
    percentage = cfg.active.percentage
    cloud_partitions = cfg.active.cloud_partitions

    model_version = cfg.active.model_version
    selection_version = cfg.active.selection_version

    dataset = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=experiment.info,
                              num_clouds=cfg.train.dataset_size, al_experiment=True, selection_mode=True)

    if seed:
        selector = get_selector(selection_objects=cloud_partitions, criterion='Random',
                                dataset_path=cfg.ds.path, project_name=experiment.info,
                                cloud_paths=dataset.cloud_files, device=device, cfg=cfg)
        selection_data = selector.select(dataset=dataset, percentage=percentage)
        selection, normal_metric_statistics, weighted_metric_statistics = selection_data
    else:
        selector = get_selector(selection_objects=cloud_partitions, criterion=criterion,
                                dataset_path=cfg.ds.path, project_name=experiment.info,
                                cloud_paths=dataset.cloud_files, device=device, cfg=cfg)

        # Load the selection from W&B
        selection_artifact = cfg.active.selection if cfg.active.selection is not None else \
            f'{experiment.selection}:{selection_version}'
        selection = load_artifact(selection_artifact)
        selector.load_voxel_selection(selection, dataset)

        # Load the model from W&B
        model_artifact = cfg.active.model if cfg.active.model is not None else \
            f'{experiment.model}:{model_version}'
        model = load_artifact(model_artifact, device=device)
        model_state_dict = model['model_state_dict']
        model = get_model(cfg=cfg, device=device)
        model.load_state_dict(model_state_dict)

        # Select the voxels to be labeled
        selection_data = selector.select(dataset=dataset, model=model, percentage=percentage)
        selection, normal_metric_statistics, weighted_metric_statistics = selection_data

    # Save the selection to W&B
    log_selection(selection=selection, selection_name=experiment.selection)

    if normal_metric_statistics is not None:
        log.info(f'Normal metric statistics')
        log_selection_metric_statistics(cfg=cfg,
                                        metric_statistics=normal_metric_statistics,
                                        metric_statistics_name=experiment.metric_stats,
                                        weighted=False)
    if weighted_metric_statistics is not None:
        log.info(f'Weighted metric statistics')
        log_selection_metric_statistics(cfg=cfg,
                                        metric_statistics=weighted_metric_statistics,
                                        metric_statistics_name=experiment.weighted_metric_stats,
                                        weighted=True)

    # Log the results of the selection to W&B
    selector.load_voxel_selection(voxel_selection=selection, dataset=dataset)
    log_dataset_statistics(cfg=cfg, dataset=dataset)

    scan = LaserScan(label_map=cfg.ds.learning_map,
                     color_map=cfg.ds.color_map_train,
                     colorize=True)
    log_most_labeled_sample(dataset=dataset, laser_scan=scan)


def load_artifact(artifact: str, device: torch.device = torch.device('cpu')):
    artifact_dir = wandb.use_artifact(artifact).download()
    data = torch.load(os.path.join(artifact_dir, f'{artifact.split("/")[-1].split(":")[0]}.pt'), map_location=device)
    return data
