import torch
import numpy as np

from .base_selector import Selector
from .voxel_selector import VoxelSelector
from .superpoint_selector import SuperpointSelector


def get_selector(selection_objects: str, criterion: str, dataset_path: str, project_name: str,
                 cloud_paths: np.ndarray, device: torch.device, batch_size: int) -> Selector:
    if selection_objects == 'Voxels':
        return VoxelSelector(dataset_path, project_name, cloud_paths, device, criterion, batch_size)
    elif selection_objects == 'Superpoints':
        return SuperpointSelector(dataset_path, project_name, cloud_paths, device, criterion, batch_size)
    else:
        raise ValueError(f'Unknown selection_objects: {selection_objects}')
