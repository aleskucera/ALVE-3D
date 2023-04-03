import torch
import numpy as np

from .voxel_selectors import RandomVoxelSelector, AverageEntropyVoxelSelector, \
    ViewpointVarianceVoxelSelector, EpistemicUncertaintyVoxelSelector
from .superpoint_selector import RandomSuperpointSelector


def get_selector(selector_type: str, dataset_path: str, cloud_paths: np.ndarray,
                 device: torch.device, dataset_percentage: float = 10):
    """ Get the selector function

    :param selector_type: The type of selector to use (random_samples, entropy_samples, random_voxels, entropy_voxels)
    :param dataset_path: The path to the dataset
    :param cloud_paths: The paths to the clouds in the dataset
    :param device: The device to use for the selection
    :param dataset_percentage: The percentage of the dataset to select (default: 10)
    """

    if selector_type == 'random_voxels':
        return RandomVoxelSelector(dataset_path, cloud_paths, device, dataset_percentage)
    elif selector_type == 'average_entropy_voxels':
        return AverageEntropyVoxelSelector(dataset_path, cloud_paths, device, dataset_percentage)
    elif selector_type == 'viewpoint_variance_voxels':
        return ViewpointVarianceVoxelSelector(dataset_path, cloud_paths, device, dataset_percentage)
    elif selector_type == 'epistemic_uncertainty_voxels':
        return EpistemicUncertaintyVoxelSelector(dataset_path, cloud_paths, device, dataset_percentage)

    elif selector_type == 'random_superpoints':
        return RandomSuperpointSelector(dataset_path, cloud_paths, device, dataset_percentage)
    else:
        raise ValueError(f'Unknown selector: {selector_type}')
