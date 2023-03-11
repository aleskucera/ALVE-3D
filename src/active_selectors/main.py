import torch
from typing import Iterable

from .voxel_selectors import RandomVoxelSelector, STDVoxelSelector
from .sample_selectors import RandomSampleSelector, EntropySampleSelector


def get_selector(selector_type: str, dataset_path: str, sequences: Iterable[int], device: torch.device,
                 dataset_percentage: float = 10):
    """ Get the selector function

    :param selector_type: The type of selector to use (random_samples, entropy_samples, random_voxels, std_voxels)
    :param dataset_path: The path to the dataset
    :param sequences: The sequences of the dataset to use
    :param device: The device to use for the selection
    :param dataset_percentage: The percentage of the dataset to select (default: 10)
    """

    # Whole sample selectors
    if selector_type == 'random_samples':
        return RandomSampleSelector(dataset_path, sequences, device, dataset_percentage)
    elif selector_type == 'entropy_samples':
        return EntropySampleSelector(dataset_path, sequences, device, dataset_percentage)

    # Voxel selectors
    elif selector_type == 'random_voxels':
        return RandomVoxelSelector(dataset_path, sequences, device, dataset_percentage)
    elif selector_type == 'std_voxels':
        return STDVoxelSelector(dataset_path, sequences, device, dataset_percentage)

    # Superpoint selectors
    elif selector_type == 'random_superpoints':
        raise NotImplementedError
    elif selector_type == 'entropy_superpoints':
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown selector: {selector_type}')
