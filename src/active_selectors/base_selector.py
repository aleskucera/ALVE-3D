import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

from .base_cloud import Cloud


class Selector(object):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device):
        self.device = device
        self.cloud_paths = cloud_paths
        self.dataset_path = dataset_path

        self.clouds = []
        self.num_voxels = 0
        self.voxels_labeled = 0

    def _initialize(self) -> None:
        raise NotImplementedError

    def select(self, dataset: Dataset, model: nn.Module = None, percentage: float = 0.5) -> dict:
        raise NotImplementedError

    def get_cloud(self, cloud_path: str) -> Cloud:
        for cloud in self.clouds:
            if cloud.path == cloud_path or cloud_path in cloud.path:
                return cloud

    def get_selection_size(self, dataset: Dataset, percentage: float) -> int:
        num_voxels = 0
        for cloud in self.clouds:
            voxel_mask = dataset.get_voxel_mask(cloud.path, cloud.size)
            num_voxels += np.sum(voxel_mask)
        return int(num_voxels * percentage / 100)

    def load_voxel_selection(self, voxel_selection: dict, dataset: Dataset = None) -> None:
        for cloud_name, label_mask in voxel_selection.items():
            cloud = self.get_cloud(cloud_name)
            voxels = torch.nonzero(label_mask).squeeze(1)
            cloud.label_voxels(voxels, dataset)

    def _map_model_predictions(self, model: nn.Module, dataset: Dataset, mc_dropout: bool) -> None:
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for cloud in self.clouds:
                indices = np.where(dataset.cloud_map == cloud.path)[0]
                for i in tqdm(indices, desc='Mapping model predictions to voxels'):
                    proj_image, proj_voxel_map, cloud_path = dataset.get_item(i)
                    proj_image = torch.from_numpy(proj_image).type(torch.float32).unsqueeze(0).to(self.device)
                    proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long)

                    sample_voxel_map = proj_voxel_map.flatten()
                    valid = (sample_voxel_map != -1)
                    sample_voxel_map = sample_voxel_map[valid]

                    if not mc_dropout:
                        model_output = model(proj_image)
                        model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)
                        model_output = model_output[valid]
                    else:
                        model_output = torch.zeros((0,), dtype=torch.float32, device=self.device)
                        for j in range(5):
                            model_output_it = model(proj_image)
                            model_output_it = model_output_it.flatten(start_dim=2).permute(0, 2, 1)
                            model_output_it = model_output_it[:, valid, :]
                            model_output = torch.cat((model_output, model_output_it), dim=0)

                    cloud.add_predictions(model_output.cpu(), sample_voxel_map, mc_dropout=mc_dropout)
