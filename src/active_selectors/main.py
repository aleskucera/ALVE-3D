import torch
import numpy as np

from .voxel_selectors import RandomVoxelSelector
from .sample_selectors import RandomSampleSelector, EntropySampleSelector


def get_selector(selector_type: str, dataset_path: str, cloud_paths: np.ndarray,
                 device: torch.device, dataset_percentage: float = 10):
    """ Get the selector function

    :param selector_type: The type of selector to use (random_samples, entropy_samples, random_voxels, entropy_voxels)
    :param dataset_path: The path to the dataset
    :param cloud_ids: The cloud ids of the dataset
    :param sequence_map: The sequence map of the dataset
    :param device: The device to use for the selection
    :param dataset_percentage: The percentage of the dataset to select (default: 10)
    """

    # if selector_type == 'random_samples':
    #     return RandomSampleSelector(dataset_path, device, dataset_percentage)
    # elif selector_type == 'entropy_samples':
    #     return EntropySampleSelector(dataset_path, device, dataset_percentage)
    if selector_type == 'random_voxels':
        return RandomVoxelSelector(dataset_path, cloud_paths, device, dataset_percentage)
    elif selector_type == 'entropy_voxels':
        return ViewpointEntropyVoxelSelector(dataset_path, cloud_paths, device, dataset_percentage)
    else:
        raise ValueError(f'Unknown selector: {selector_type}')
