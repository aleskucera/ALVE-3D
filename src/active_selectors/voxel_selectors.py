import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset


class VoxelCloud(object):
    """ An object that represents a global cloud in a dataset sequence. The voxels are areas of space that can contain
    multiple points from different frames. The cloud is a collection of these voxels. This class is used to store
    values for each voxel in the cloud. The values are used to select the most informative voxels for labeling.

    :param size: The number of voxels in the cloud
    :param cloud_id: The id of the cloud (used to identify the cloud in the dataset, unique for each cloud in dataset)
    :param sequence: The sequence of the cloud
    :param seq_cloud_id: The id of the cloud in the sequence (unique for each cloud in sequence)
    :param device: The device where the tensors are stored
    """

    def __init__(self, size: int, cloud_id: int, sequence: int, seq_cloud_id: int, device: torch.device):
        self.size = size
        self.id = cloud_id
        self.device = device
        self.sequence = sequence
        self.seq_cloud_id = seq_cloud_id

        self.voxel_map = torch.zeros((0,), dtype=torch.int32, device=device)
        self.predictions = torch.zeros((0,), dtype=torch.float32, device=device)
        self.labeled_voxels = torch.zeros(size, dtype=torch.bool, device=device)

    @property
    def num_classes(self):
        if self.predictions.shape[0] == 0:
            return None
        return self.predictions.shape[1]

    def add_predictions(self, predictions: torch.Tensor, voxel_map: torch.Tensor) -> None:
        """ Add model predictions to the cloud.

        :param predictions: The model predictions to add to the cloud with shape (N, C), where N is the
                            number of inputs (pixels) and C is the number of classes.
        :param voxel_map: The voxel map with shape (N,), where N is the number of sample inputs (pixels).
                          The voxel map maps each prediction to a voxel in the cloud.
        """

        # Get the indices of the unlabeled voxels
        unlabeled_voxels = torch.nonzero(~self.labeled_voxels).squeeze(1)

        # Remove the values of the voxels that are already labeled
        indices = torch.where(torch.isin(voxel_map, unlabeled_voxels))
        unlabeled_predictions = predictions[indices]
        unlabeled_voxel_map = voxel_map[indices]

        # concatenate new values to existing values
        self.predictions = torch.cat((self.predictions, unlabeled_predictions), dim=0)
        self.voxel_map = torch.cat((self.voxel_map, unlabeled_voxel_map), dim=0)

    def label_voxels(self, voxels: torch.Tensor, dataset: Dataset):
        """ Label the given voxels in the dataset."""
        self.labeled_voxels[voxels] = True
        dataset.label_voxels(voxels.cpu().numpy(), self.sequence, self.seq_cloud_id)

    def get_viewpoint_entropies(self):
        # Initialize the output tensor with NaNs
        viewpoint_entropies = torch.full((self.size,), float('nan'), dtype=torch.float32, device=self.device)

        # Sort the model outputs ascending by voxel map
        order = torch.argsort(self.voxel_map)
        predictions = self.predictions[order]

        # Split the predictions to a list of tensors where each tensor contains a set of predictions for a voxel
        unique_voxels, num_views = torch.unique(self.voxel_map, return_counts=True)
        prediction_sets = torch.split(predictions, num_views.tolist())

        entropies = torch.tensor([], device=self.device)
        for prediction_set in prediction_sets:
            entropy = self.calculate_entropy(prediction_set).unsqueeze(0)
            entropies = torch.cat((entropies, entropy), dim=0)

        viewpoint_entropies[unique_voxels.type(torch.long)] = entropies
        return self._append_mapping(viewpoint_entropies)

    @staticmethod
    def calculate_entropy(probability_distribution_set: torch.Tensor):
        # Calculate mean distribution over all viewpoints
        mean_distribution = torch.mean(probability_distribution_set, dim=0)

        # Calculate entropy of mean distribution
        entropy = torch.sum(-mean_distribution * torch.log(mean_distribution))
        return entropy

    def _append_mapping(self, values: torch.Tensor):
        """ Append the voxel indices and cloud ids to the values and return the result.

        :param values: The values to append the mapping to
        :return: A tuple containing the values, voxel indices and cloud ids
        """

        nan_indices = torch.isnan(values)
        filtered_values = values[~nan_indices]
        filtered_voxel_indices = torch.arange(self.size, device=self.device)[~nan_indices]
        filtered_cloud_ids = torch.full((self.size,), self.id, dtype=torch.int32, device=self.device)[~nan_indices]
        return filtered_values, filtered_voxel_indices, filtered_cloud_ids

    def __len__(self):
        return self.size

    def __str__(self):
        ret = f'\nVoxelCloud:\n' \
              f'\t - Cloud ID = {self.id}, \n' \
              f'\t - Sequence = {self.sequence}, \n' \
              f'\t - Cloud ID relative to sequence = {self.seq_cloud_id}, \n' \
              f'\t - Number of voxels in cloud = {self.size}\n' \
              f'\t - Number of model predictions = {self.predictions.shape[0]}\n'

        if self.predictions.shape[0] > 0:
            ret += f'\t - Number of semantic classes = {self.predictions.shape[1]}\n'

        ret += f'\t - Number of already labeled voxels = {torch.sum(self.labeled_voxels)}\n'
        return ret


class BaseVoxelSelector:
    """ Base class for voxel selectors """

    def __init__(self, dataset_path: str, seq_cloud_ids: np.ndarray, sequence_map: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        self.device = device
        self.dataset_path = dataset_path
        self.dataset_percentage = dataset_percentage

        self.cloud_ids = torch.arange(len(seq_cloud_ids), dtype=torch.long, device=device)
        self.seq_cloud_ids = torch.from_numpy(seq_cloud_ids).type(torch.long).to(device)
        self.sequence_map = torch.from_numpy(sequence_map).type(torch.long).to(device)

        self.clouds = []
        self.num_voxels = 0
        self.voxels_labeled = 0

        self._initialize()

    def _initialize(self):
        """ Create a list of VoxelCloud objects and count the total number of voxels in the dataset """

        for cloud_id, seq_cloud_id, sequence in zip(self.cloud_ids, self.seq_cloud_ids, self.sequence_map):
            # Get all cloud files for the current sequence
            clouds_dir = os.path.join(self.dataset_path, 'sequences', f'{sequence:02d}', 'global_clouds')
            seq_cloud_files = sorted([os.path.join(clouds_dir, cloud) for cloud in os.listdir(clouds_dir)])

            # Create a VoxelCloud object for each cloud file
            cloud_file = seq_cloud_files[seq_cloud_id]
            with h5py.File(cloud_file, 'r') as f:
                num_voxels = f['points'].shape[0]
                self.num_voxels += num_voxels
                self.clouds.append(VoxelCloud(num_voxels, cloud_id, sequence, seq_cloud_id, self.device))

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled """

        raise NotImplementedError

    def is_finished(self):
        return self.voxels_labeled == self.num_voxels

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

    def _get_cloud(self, sequence: int, seq_cloud_id: int):
        for cloud in self.clouds:
            if cloud.sequence == sequence and cloud.seq_cloud_id == seq_cloud_id:
                return cloud


class RandomVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_ids: np.ndarray, sequence_map: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_ids, sequence_map, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled """

        # Calculate, how many voxels should be labeled
        num_classes = 20
        selection_size = int(self.num_voxels * self.dataset_percentage / 100)

        for i in tqdm(range(dataset.get_length()),
                      desc='Simulating model forward pass and mapping predictions to voxels'):
            _, proj_voxel_map, sequence, seq_cloud_id = dataset.get_item(i)
            proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long).to(self.device)

            # Get the model output for the sample
            model_output = torch.rand((1, num_classes, dataset.proj_H, dataset.proj_W),
                                      dtype=torch.float32, device=self.device)
            model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)
            voxel_map = proj_voxel_map.flatten()

            # Find the global cloud to which the sample belongs and map the values to the voxels
            cloud = self._get_cloud(sequence, seq_cloud_id)
            cloud.add_predictions(model_output, voxel_map)

        # Generate a random values for each inactive voxel
        values = torch.tensor([], dtype=torch.float32, device=self.device)
        voxel_map = torch.tensor([], dtype=torch.long, device=self.device)
        cloud_map = torch.tensor([], dtype=torch.long, device=self.device)

        for cloud in tqdm(self.clouds, desc='Calculating mean values for generated random values'):
            cloud_values, cloud_voxel_map, cloud_cloud_map = cloud.get_viewpoint_entropies()
            values = torch.cat((values, cloud_values))
            voxel_map = torch.cat((voxel_map, cloud_voxel_map))
            cloud_map = torch.cat((cloud_map, cloud_cloud_map))

        # Select the samples with the highest values
        selected_voxels, selected_clouds = self.select_samples(values, voxel_map, cloud_map,
                                                               selection_size, criterion='highest')

        # Activate the selected voxels
        for cloud in self.clouds:
            cloud.label_voxels(selected_voxels[selected_clouds == cloud.id], dataset)


class ViewpointEntropyVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_ids: np.ndarray, sequence_map: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_ids, sequence_map, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled """

        # Calculate, how many voxels should be labeled
        selection_size = int(self.num_voxels * self.dataset_percentage / 100)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for i in tqdm(range(dataset.get_length()), desc='Mapping model output values to voxels'):
                proj_image, proj_voxel_map, sequence, seq_cloud_id = dataset.get_item(i)
                proj_image = torch.from_numpy(proj_image).type(torch.float32).unsqueeze(0).to(self.device)
                proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long).to(self.device)

                # Get the model output for the sample
                model_output = model(proj_image)

                # Change the shape of the output to match the shape of the flattened voxel map in 0th dimension
                model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)
                voxel_map = proj_voxel_map.flatten()

                # Find the global cloud to which the sample belongs and map the values to the voxels
                cloud = self._get_cloud(sequence, seq_cloud_id)
                cloud.add_predictions(model_output, voxel_map)

            # Calculate the viewpoint entropies for each voxel
            viewpoint_entropies = torch.tensor([], dtype=torch.float32, device=self.device)
            voxel_map = torch.tensor([], dtype=torch.long, device=self.device)
            cloud_map = torch.tensor([], dtype=torch.long, device=self.device)

            #
            for cloud in tqdm(self.clouds, desc='Calculating the standard deviation for each voxel'):
                entropies, cloud_voxel_map, cloud_cloud_map = cloud.get_viewpoint_entropies()
                viewpoint_entropies = torch.cat((viewpoint_entropies, entropies))
                voxel_map = torch.cat((voxel_map, cloud_voxel_map))
                cloud_map = torch.cat((cloud_map, cloud_cloud_map))

            # Select the samples with the highest viewpoint entropy
            selected_voxels, selected_clouds = self.select_samples(viewpoint_entropies, voxel_map, cloud_map,
                                                                   selection_size, criterion='highest')

            # Activate the selected voxels
            for cloud in self.clouds:
                cloud.label_voxels(selected_voxels[selected_clouds == cloud.id], dataset)
