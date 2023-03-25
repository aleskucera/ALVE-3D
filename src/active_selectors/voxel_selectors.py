import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

from .voxel_cloud import VoxelCloud


class BaseVoxelSelector:
    """ Base class for voxel selectors

    :param dataset_path: Path to the dataset
    :param device: Device to use for computations
    :param dataset_percentage: Percentage of the dataset to be labeled each iteration (default: 10)
    """

    def __init__(self, dataset_path: str, cloud_paths: np.ndarray,
                 device: torch.device, dataset_percentage: float = 10):
        self.device = device
        self.dataset_path = dataset_path
        self.dataset_percentage = dataset_percentage

        self.cloud_ids = torch.arange(len(cloud_paths), dtype=torch.long, device=device)
        self.cloud_paths = cloud_paths

        self.clouds = []
        self.num_voxels = 0
        self.voxels_labeled = 0

        self._initialize()

    def _initialize(self):
        """ Create a list of VoxelCloud objects from a given sequence cloud ids and their sequence map.
        The function also computes the total number of voxels in the dataset to determine the number of
        voxels to be labeled each iteration and if the dataset is fully labeled.
        """
        print(f'Cloud paths: {self.cloud_paths}')
        for cloud_id, cloud_path in zip(self.cloud_ids, self.cloud_paths):
            with h5py.File(cloud_path, 'r') as f:
                num_voxels = f['points'].shape[0]
                label_mask = torch.tensor(f['label_mask'][...]).type(torch.bool)
                # voxel_mask = torch.tensor(f['voxel_mask'][...]).type(torch.bool)
                self.num_voxels += num_voxels
                self.clouds.append(VoxelCloud(cloud_path, num_voxels, label_mask, cloud_id))
                # print(f'Voxel mask percentage: {voxel_mask.sum() / len(voxel_mask)}')

        # # Iterate over each cloud id given and determine the file path, then create a VoxelCloud object
        # for cloud_id, seq_cloud_id, sequence in zip(self.cloud_ids, self.seq_cloud_ids, self.sequence_map):
        #     clouds_dir = os.path.join(self.dataset_path, 'sequences', f'{sequence:02d}', 'global_clouds')
        #     seq_cloud_files = sorted([os.path.join(clouds_dir, cloud) for cloud in os.listdir(clouds_dir)])
        #
        #     cloud_file = seq_cloud_files[seq_cloud_id]
        #     with h5py.File(cloud_file, 'r') as f:
        #         num_voxels = f['points'].shape[0]
        #         label_mask = torch.tensor(f['label_mask'][...]).type(torch.bool)
        #         self.num_voxels += num_voxels
        #         self.clouds.append(VoxelCloud(cloud_file, num_voxels, label_mask, cloud_id, sequence, seq_cloud_id))

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled """

        raise NotImplementedError

    def is_finished(self):
        return self.voxels_labeled == self.num_voxels

    def get_cloud(self, cloud_path: str):
        """ Get the VoxelCloud object for the given sequence and cloud id """
        for cloud in self.clouds:
            if cloud.path == cloud_path:
                return cloud


class RandomVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_paths, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled randomly, but simulate the
        model forward pass and the voxels mapping. The function executes following steps:

        1. Define the number of classes in the dataset and the number of voxels to be labeled.
        2. Iterate over the dataset and simulate the model forward pass and the voxels mapping by generating
           random values with the same shape as the model output.
        3. Change the shape of the model output to (num_voxels, num_classes) and flatten the distances and
           voxel map to (num_voxels,).
        4. Remove the voxels that has NaN values in the voxel map which means that on this pixel was empty due to
           the projection or the label has been mapped to the ignore class.
        5. Add the randomly generated predictions with the voxel map and distances to the VoxelCloud object.
        6. Calculate the Viewpoint Entropy for each voxel in the VoxelCloud. (Not necessary, but
           this step is used to test the voxel selection process)
        7. Select the voxels with the highest Viewpoint Entropy and label them.

        :param dataset: Dataset object
        :param model: Model object (not used)
        """

        # Define constants
        num_classes = 18
        selection_size = int(self.num_voxels * self.dataset_percentage / 100)

        for i in tqdm(range(dataset.get_full_length()), desc='Simulating model forward pass and voxels mapping'):
            _, proj_distances, proj_voxel_map, cloud_path = dataset.get_item(i)
            proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long)
            proj_distances = torch.from_numpy(proj_distances).type(torch.float32)

            # Simulate the model forward pass
            model_output = torch.rand((1, num_classes, dataset.proj_H, dataset.proj_W), dtype=torch.float32)
            model_output = torch.softmax(model_output, dim=1)
            model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)
            distances = proj_distances.flatten()
            voxel_map = proj_voxel_map.flatten()

            # Remove the voxels where voxel map is -1 or 0 (empty pixel or ignore class)
            valid = (voxel_map != -1)
            model_output, distances, voxel_map = model_output[valid], distances[valid], voxel_map[valid]

            # Find the VoxelCloud object and add the generated predictions
            cloud = self.get_cloud(cloud_path)
            cloud.add_predictions(model_output, distances, voxel_map)

        # Calculate the viewpoint entropies for each voxel
        values = torch.tensor([], dtype=torch.float32, device=self.device)
        voxel_map = torch.tensor([], dtype=torch.long, device=self.device)
        cloud_map = torch.tensor([], dtype=torch.long, device=self.device)

        for cloud in tqdm(self.clouds, desc='Calculating mean values for generated random values'):
            cloud_values, cloud_voxel_map, cloud_cloud_map = cloud.get_viewpoint_entropies()
            values = torch.cat((values, cloud_values.to(self.device)))
            voxel_map = torch.cat((voxel_map, cloud_voxel_map.to(self.device)))
            cloud_map = torch.cat((cloud_map, cloud_cloud_map.to(self.device)))

        # Select the voxels with the highest values
        order = torch.argsort(values, descending=True)
        voxel_map, cloud_map = voxel_map[order], cloud_map[order]
        print(f'Number of voxels: {len(voxel_map)}')
        selected_voxels, cloud_map = voxel_map[:selection_size], cloud_map[:selection_size]
        print(f'Number of selected voxels: {len(selected_voxels)}')

        # Label the selected voxels
        for cloud in self.clouds:
            cloud.label_voxels(selected_voxels[cloud_map == cloud.id], dataset)
            cloud.reset()


class ViewpointEntropyVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_paths, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the voxels to be labeled by calculating the Viewpoint Entropy for each voxel and
        selecting the voxels with the highest Viewpoint Entropy. The function executes following steps:

        1. Define the number of voxels to be labeled.
        2. Iterate over the dataset and get the model output for each sample.
        3. Change the shape of the model output to (num_voxels, num_classes) and flatten the distances and
              voxel map to (num_voxels,).
        4. Remove the voxels that has NaN values in the voxel map which means that on this pixel was empty due to
              the projection or the label has been mapped to the ignore class.
        5. Add the model output with the voxel map and distances to the VoxelCloud object.
        6. Calculate the Viewpoint Entropy for each voxel in the VoxelCloud.
        7. Select the voxels with the highest Viewpoint Entropy and label them.

        :param dataset: Dataset object
        :param model: Model based on which the voxels will be selected
        """

        # Calculate, how many voxels should be labeled
        selection_size = int(self.num_voxels * self.dataset_percentage / 100)

        clouds = np.unique(dataset.cloud_map)
        all_viewpoint_entropies = torch.tensor([], dtype=torch.float32)
        all_voxel_map = torch.tensor([], dtype=torch.long)
        all_cloud_map = torch.tensor([], dtype=torch.long)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for cloud in clouds:
                indices = np.where(dataset.cloud_map == cloud)[0]
                cloud_obj = self.get_cloud(cloud)
                for i in tqdm(indices, desc='Mapping model output values to voxels'):
                    proj_image, proj_distances, proj_voxel_map, cloud_path = dataset.get_item(i)
                    proj_image = torch.from_numpy(proj_image).type(torch.float32).unsqueeze(0).to(self.device)
                    proj_distances = torch.from_numpy(proj_distances).type(torch.float32)
                    proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long)

                    # Forward pass
                    model_output = model(proj_image)

                    # Change the shape of the model output to (num_voxels, num_classes)
                    model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)
                    distances = proj_distances.flatten()
                    voxel_map = proj_voxel_map.flatten()

                    # Remove the voxels where voxel map is NaN (empty pixel or ignore class)
                    valid = (voxel_map != -1)
                    model_output, distances, voxel_map = model_output[valid], distances[valid], voxel_map[valid]

                    cloud_obj.add_predictions(model_output.cpu(), distances, voxel_map)

                entropies, cloud_voxel_map, cloud_cloud_map = cloud_obj.get_viewpoint_entropies()
                all_viewpoint_entropies = torch.cat((all_viewpoint_entropies, entropies))
                all_voxel_map = torch.cat((all_voxel_map, cloud_voxel_map))
                all_cloud_map = torch.cat((all_cloud_map, cloud_cloud_map))

                print(f'Voxel map shape {all_voxel_map.shape}')
                print(f'Cloud map shape {all_cloud_map.shape}')
                print(f'Viewpoint entropies shape {all_viewpoint_entropies.shape}')

                cloud_obj.reset()

            # Select the samples with the highest viewpoint entropy
            order = torch.argsort(viewpoint_entropies, descending=True)
            voxel_map, cloud_map = voxel_map[order], cloud_map[order]
            selected_voxels, cloud_map = voxel_map[:selection_size], cloud_map[:selection_size]

            # Label the selected voxels
            for cloud in self.clouds:
                cloud.label_voxels(selected_voxels[cloud_map == cloud.id], dataset)
