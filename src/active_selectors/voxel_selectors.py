import os
import time
from typing import Iterable

import h5py
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset


class VoxelCloud(object):
    def __init__(self, size: int, cloud_id: int, sequence: int, seq_cloud_id: int, device: torch.device):
        self.size = size
        self.device = device
        self.id = cloud_id
        self.sequence = sequence
        self.seq_cloud_id = seq_cloud_id

        self.values = torch.zeros((0,), dtype=torch.float32, device=device)
        self.voxel_map = torch.zeros((0,), dtype=torch.int32, device=device)

        self.labeled_voxels = torch.zeros(size, dtype=torch.bool, device=device)

    def add_values(self, values: torch.Tensor, voxel_map: torch.Tensor):
        unlabeled_voxels = torch.where(self.labeled_voxels == 0)[0]

        indices = torch.where(torch.isin(voxel_map, unlabeled_voxels))
        inactive_values = values[indices]
        inactive_voxel_map = voxel_map[indices]

        # concatenate new values to existing values
        self.values = torch.cat((self.values, inactive_values), dim=0)
        self.voxel_map = torch.cat((self.voxel_map, inactive_voxel_map), dim=0)

    def label_voxels(self, voxels: torch.Tensor, dataset: Dataset):
        self.labeled_voxels[voxels] = True
        dataset.label_voxels(voxels.cpu().numpy(), self.sequence, self.seq_cloud_id)

    def get_std_values_2(self):
        values = torch.full((self.size,), float('nan'), dtype=torch.float32, device=self.device)
        for voxel in tqdm(torch.unique(self.voxel_map)):
            voxel_values = self.values[self.voxel_map == voxel]
            voxel_std = torch.std(voxel_values)
            values[voxel] = voxel_std
        return self._append_mapping(values)

    def get_std_values(self):
        values = torch.full((self.size,), float('nan'), dtype=torch.float32, device=self.device)

        # Sort the values ascending by voxel map
        sorted_indices = torch.argsort(self.voxel_map)
        sorted_values = self.values[sorted_indices]

        # Split the values to a multidimensional tensor, where each row contains the values of a voxel
        unique_voxels, counts = torch.unique(self.voxel_map, return_counts=True)
        voxel_values = torch.split(sorted_values, counts.tolist())

        # Calculate the standard deviation of each voxel and assign it to the corresponding voxel
        voxel_stds = torch.tensor([torch.std(v) for v in voxel_values], device=self.device)
        values[unique_voxels] = voxel_stds

        return self._append_mapping(values)

    def get_random_values(self):
        values = torch.full((self.size,), float('nan'), dtype=torch.float32, device=self.device)
        values[~self.labeled_voxels] = torch.rand((self.size - self.labeled_voxels.sum(),), device=self.device)
        return self._append_mapping(values)

    def _append_mapping(self, values: torch.Tensor):
        nan_indices = torch.isnan(values)
        filtered_values = values[~nan_indices]
        filtered_voxel_indices = torch.arange(self.size, device=self.device)[~nan_indices]
        filtered_cloud_ids = torch.full((self.size,), self.id, dtype=torch.int32, device=self.device)[~nan_indices]
        return filtered_values, filtered_voxel_indices, filtered_cloud_ids

    def __len__(self):
        return self.size

    def __str__(self):
        return f'\nVoxelCloud:' \
               f'\t - id = {self.id}, \n' \
               f'\t - sequence = {self.sequence}, \n' \
               f'\t - seq_cloud_id = {self.seq_cloud_id}, \n' \
               f'\t - size = {self.size}\n' \
               f'\t - values shape= {self.values.shape}, \n' \
               f'\t - values_mask shape = {self.voxel_map.shape}, \n'

    def __repr__(self):
        return self.__str__()


class BaseVoxelSelector:
    def __init__(self, dataset_path: str, sequences: Iterable[int], device: torch.device,
                 dataset_percentage: float = 10):
        self.device = device
        self.sequences = sequences
        self.dataset_path = dataset_path
        self.dataset_percentage = dataset_percentage

        self.clouds = []
        self.num_voxels = 0
        self.voxels_labeled = 0

        self._initialize()

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled """

        raise NotImplementedError

    def is_finished(self):
        return self.voxels_labeled == self.num_voxels

    def map_model_outputs_to_clouds(self, dataset: Dataset, model: nn.Module):
        """ Iterate over the dataset and map the model outputs to the corresponding voxels """

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            # Iterate over the dataset (len(dataset) would give only the already labeled samples in the dataset)
            for i in tqdm(range(dataset.get_true_length()), desc='Mapping model output values to voxels'):
                # Parse the data (get_item() is also special function for selector classes)
                proj_image, proj_label, proj_voxel_map, sequence, seq_cloud_id = dataset.get_item(i)
                proj_image, proj_label = proj_image.to(self.device), proj_label.to(self.device)
                proj_voxel_map = proj_voxel_map.to(self.device)

                # Get the model output for the sample
                model_output = torch.rand(proj_label.shape, dtype=torch.float32, device=self.device)
                # model_output = model(proj_image.unsqueeze(0))
                # TODO: Dimension problem
                model_output = model_output.flatten()
                voxel_map = proj_voxel_map.flatten().type(torch.long)

                # Find the global cloud to which the sample belongs and map the values to the voxels
                cloud = self._get_cloud(sequence, seq_cloud_id)
                cloud.add_values(model_output, voxel_map)

    @staticmethod
    def select_samples(values: torch.Tensor, voxel_map: torch.Tensor, cloud_map: torch.Tensor,
                       selection_size: int, criterion: str):
        """ Select the samples with the highest values """

        # Sort the values and their mappings
        if criterion == 'lowest':
            order = torch.argsort(values)
        elif criterion == 'highest':
            order = torch.argsort(values, descending=True)
        else:
            raise ValueError(f'Invalid criterion: {criterion}')

        voxel_map = voxel_map[order]
        cloud_map = cloud_map[order]

        # Select the voxels with the highest values
        selected_voxels = voxel_map[:selection_size]
        selected_clouds = cloud_map[:selection_size]

        return selected_voxels, selected_clouds

    def _initialize(self):
        """ Create a list of VoxelCloud objects and count the total number of voxels in the dataset """

        cloud_id = 0  # unique id for each cloud

        for sequence in self.sequences:

            # Get all cloud files for the current sequence
            clouds_dir = os.path.join(self.dataset_path, 'sequences', f'{sequence:02d}', 'global_clouds')
            seq_cloud_files = [os.path.join(clouds_dir, cloud) for cloud in os.listdir(clouds_dir) if
                               cloud.endswith('.h5')]
            seq_cloud_files = sorted(seq_cloud_files)

            # Create a VoxelCloud object for each cloud file
            for seq_cloud_id, cloud_file in enumerate(seq_cloud_files):
                with h5py.File(cloud_file, 'r') as f:
                    num_voxels = f['points'].shape[0]
                    self.num_voxels += num_voxels
                    self.clouds.append(VoxelCloud(num_voxels, cloud_id, sequence, seq_cloud_id, self.device))
                    cloud_id += 1

    def _get_cloud(self, sequence: int, seq_cloud_id: int):
        for cloud in self.clouds:
            if cloud.sequence == sequence and cloud.seq_cloud_id == seq_cloud_id:
                return cloud


class RandomVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, sequences: Iterable[int], device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, sequences, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled """

        # Calculate, how many voxels should be labeled
        selection_size = int(self.num_voxels * self.dataset_percentage / 100)

        # Generate a random values for each inactive voxel
        means, voxel_map, cloud_map = [], [], []
        for i in tqdm(range(len(self.clouds)), desc='Generating random values for each voxel'):
            if i != 0:
                mean_values, voxels, cloud_ids = self.clouds[i].get_random_values()

                means.append(mean_values)
                voxel_map.append(voxels)
                cloud_map.append(cloud_ids)

        # Concatenate the values and their mappings
        means = torch.cat(means)
        voxel_map = torch.cat(voxel_map)
        cloud_map = torch.cat(cloud_map)

        # Select the samples with the highest values
        selected_voxels, selected_clouds = self.select_samples(means, voxel_map, cloud_map, selection_size,
                                                               criterion='highest')

        # # Activate the selected voxels
        # for cloud in self.clouds:
        #     cloud.label_voxels(selected_voxels[selected_clouds == cloud.id], dataset)


class STDVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, sequences: Iterable[int], device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, sequences, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled """

        # Calculate, how many voxels should be labeled
        selection_size = int(self.num_voxels * self.dataset_percentage / 100)

        # Map the model outputs to the voxels (add outputs from different viewpoints)
        self.map_model_outputs_to_clouds(dataset, model)

        # Get the calculated values and their mappings
        means, voxel_map, cloud_map = [], [], []
        for i in tqdm(range(len(self.clouds)), desc='Calculating values for each voxel'):
            if i != 0:
                # TODO: Resolve val clouds
                mean_values, voxels, cloud_ids = self.clouds[i].get_std_values()

                means.append(mean_values)
                voxel_map.append(voxels)
                cloud_map.append(cloud_ids)

        # Concatenate the values and their mappings
        means = torch.cat(means)
        voxel_map = torch.cat(voxel_map)
        cloud_map = torch.cat(cloud_map)

        # Select the samples with the highest values
        selected_voxels, selected_clouds = self.select_samples(means, voxel_map, cloud_map, selection_size,
                                                               criterion='highest')

        # # Activate the selected voxels
        # for cloud in self.clouds:
        #     cloud.label_voxels(selected_voxels[selected_clouds == cloud.id], dataset)
