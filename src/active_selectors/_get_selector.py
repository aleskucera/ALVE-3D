import torch
from typing import Iterable

from .voxel_selectors import RandomVoxelSelector, STDVoxelSelector


def get_selector(selector_type: str, dataset_path: str, sequences: Iterable[int], device: torch.device):
    if selector_type == 'random':
        return RandomVoxelSelector(dataset_path, sequences, device)
    elif selector_type == 'std':
        return STDVoxelSelector(dataset_path, sequences, device)
    else:
        raise ValueError(f'Unknown selector: {selector_type}')
