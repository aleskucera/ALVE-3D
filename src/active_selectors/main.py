import torch
import numpy as np

from .base_selector import Selector
from .voxel_selector import VoxelSelector
from .superpoint_selector import SuperpointSelector


def get_selector(selection_objects: str, criterion: str,
                 dataset_path: str, cloud_paths: np.ndarray, device: torch.device) -> Selector:
    if selection_objects == 'voxels':
        return VoxelSelector(dataset_path, cloud_paths, device, criterion)
    elif selection_objects == 'superpoints':
        return SuperpointSelector(dataset_path, cloud_paths, device, criterion)
    else:
        raise ValueError(f'Unknown selection_objects: {selection_objects}')
