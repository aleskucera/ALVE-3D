import os
from collections import OrderedDict

import torch
import wandb
import shutil
import logging
from typing import Any

from src.utils.experiment import Experiment

log = logging.getLogger(__name__)


def error_alert(experiment: Experiment):
    log.error(f'Experiment failed: {experiment}')
    wandb.alert(
        title='Experiment failed',
        text=f'Experiment info: {experiment}',
        level=wandb.AlertLevel.ERROR
    )


def success_alert(experiment: Experiment):
    log.info(f'Experiment finished successfully: {experiment}')
    wandb.alert(
        title='Experiment finished successfully',
        text=f'Experiment info: {experiment}',
        level=wandb.AlertLevel.INFO
    )


def pull_artifact(artifact: str, device: torch.device = torch.device('cpu')) -> Any:
    """ Pulls a data from W&B stored as an artifact. The data can be a torch.Tensor
    or a dictionary of torch.Tensors.

    There are four ways to specify the artifact:
        1. Full path: project/artifact:version -> project/artifact:version
        2. Path without version: project/artifact -> project/artifact:latest
        3. Artifact name with version: artifact:version -> current_project/artifact:version
        4. Artifact name without version: artifact -> current_project/artifact:latest

    :param artifact: The name of the artifact to be pulled. Can be full path or just the name of the artifact.
    If only the name is specified the artifact will be pulled from the current project with the latest version.
    :param device: The device to be used for the training.
    :return: The data stored in the artifact.
    """

    # Check if the wand run is initialized
    api = wandb.Api() if wandb.run is None else None

    if '/' in artifact and ':' in artifact:
        file_name = artifact.split('/')[-1].split(':')[0]
    elif ':' in artifact:
        file_name = artifact.split(':')[0]
    elif '/' in artifact:
        file_name = artifact.split('/')[-1]
        artifact = f'{artifact}:latest'
    else:
        file_name = artifact
        artifact = f'{artifact}:latest'

    try:
        if api is not None:
            log.debug('W&B run not initialized. '
                      'Using the api to pull the artifact.')
            artifact_dir = api.artifact(artifact).download()
        else:
            artifact_dir = wandb.use_artifact(artifact).download()
    except wandb.errors.CommError:
        log.warning(f'Artifact {artifact} not found in W&B.')
        return None

    data = torch.load(os.path.join(artifact_dir, f'{file_name}.pt'), map_location=device)
    log.info(f'Artifact {artifact} pulled from W&B.')
    shutil.rmtree(artifact_dir)
    return data


def push_artifact(artifact: str, data: Any, artifact_type: str, metadata: dict = None,
                  description: str = None):
    """ Pushes a data to W&B as an artifact. The data can be a torch.Tensor
    or a dictionary of torch.Tensors.

    :param artifact: The name of the artifact to be pushed.
    :param data: The data to be pushed. Can be a torch.Tensor or a dictionary of torch.Tensors.
    :param artifact_type: The type of the artifact.
    :param metadata: The metadata of the artifact.
    :param description: The description of the artifact.
    """

    assert isinstance(data, (torch.Tensor, dict, OrderedDict)), \
        'Data must be a torch.Tensor or a dictionary of torch.Tensors.'

    path = f'{artifact}.pt'
    torch.save(data, path)
    log.info(f'Pushing {artifact} to W&B.')
    artifact = wandb.Artifact(artifact,
                              type=artifact_type,
                              metadata=metadata,
                              description=description)
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    os.remove(path)
