import h5py
import torch
import numpy as np
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

        self.cloud_ids = torch.arange(len(cloud_paths), dtype=torch.long)
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
        for cloud_id, cloud_path in zip(self.cloud_ids, self.cloud_paths):
            with h5py.File(cloud_path, 'r') as f:
                num_voxels = f['points'].shape[0]
                label_mask = torch.zeros(num_voxels, dtype=torch.bool)
                self.num_voxels += num_voxels
                self.clouds.append(VoxelCloud(cloud_path, num_voxels, label_mask, cloud_id))

    def get_cloud(self, cloud_path: str):
        """ Get the VoxelCloud object for the given sequence and cloud id """
        for cloud in self.clouds:
            if cloud.path == cloud_path or cloud_path in cloud.path:
                return cloud

    def get_selection_size(self, dataset: Dataset, percentage: float):
        num_voxels = 0
        for cloud in self.clouds:
            voxel_mask = dataset.get_voxel_mask(cloud.path, cloud.size)
            num_voxels += np.sum(voxel_mask)
        return int(num_voxels * percentage / 100)

    def get_voxel_selection(self, selected_voxels: torch.Tensor, cloud_map: torch.Tensor):
        voxel_selection = dict()
        for cloud in self.clouds:
            cloud.label_voxels(selected_voxels[cloud_map == cloud.id])
            split = cloud.path.split('/')
            name = '/'.join(split[-3:])
            voxel_selection[name] = cloud.label_mask
        return voxel_selection

    def load_voxel_selection(self, voxel_selection: dict, dataset: Dataset = None):
        for cloud_name, label_mask in voxel_selection.items():
            cloud = self.get_cloud(cloud_name)
            voxels = torch.nonzero(label_mask).squeeze(1)
            cloud.label_voxels(voxels, dataset)


class RandomVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_paths, device, dataset_percentage)

    def select(self, dataset: Dataset, percentage: float = 1):

        selection_size = self.get_selection_size(dataset, percentage)

        voxels = [torch.tensor([], dtype=torch.long) for _ in range(len(self.clouds))]

        # Iterate over the scans in the dataset and get the voxels that are in the scan projection
        for i in tqdm(range(dataset.get_full_length()), desc='Getting information about voxels in the dataset'):
            _, _, proj_voxel_map, cloud_path = dataset.get_item(i)
            voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long).flatten()

            voxel_map = voxel_map[voxel_map != -1]
            vox = torch.unique(voxel_map)
            voxels[self.get_cloud(cloud_path).id] = torch.cat((voxels[self.get_cloud(cloud_path).id], vox))

        # Create a voxel map and a cloud map that maps each voxel to the cloud it belongs to
        voxel_map = torch.tensor([], dtype=torch.long, device=self.device)
        cloud_map = torch.tensor([], dtype=torch.long, device=self.device)
        for cloud in self.clouds:
            # Get the unique voxels that are in the scan projection and not labeled
            cloud_voxels = torch.unique(voxels[cloud.id]).to(self.device)
            labeled_cloud_voxels = torch.nonzero(cloud.label_mask).squeeze(1).to(self.device)
            mask = torch.isin(cloud_voxels, labeled_cloud_voxels, invert=True)
            cloud_voxels = cloud_voxels[mask]

            # Generate cloud map for the voxels
            cloud_cloud_map = torch.full((cloud_voxels.shape[0],), cloud.id, dtype=torch.long, device=self.device)

            voxel_map = torch.cat((voxel_map, cloud_voxels))
            cloud_map = torch.cat((cloud_map, cloud_cloud_map))

        # Shuffle randomly the voxel map and the cloud map and select the first selection_size voxels
        order = torch.randperm(voxel_map.shape[0], device=self.device)
        voxel_map, cloud_map = voxel_map[order], cloud_map[order]
        selected_voxels = voxel_map[:selection_size].cpu()
        selected_clouds = cloud_map[:selection_size].cpu()

        return self.get_voxel_selection(selected_voxels, selected_clouds)


class AverageEntropyVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_paths, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module, percentage: float = 1):
        """ Select the voxels to be labeled by calculating the Viewpoint Entropy for each voxel and
        selecting the voxels with the highest Viewpoint Entropy. The function executes following steps:

        1. Define the number of voxels to be labeled.
        2. Iterate over the dataset and get the model output for each sample.
        3. Change the shape of the model output to (num_voxels, num_classes) and flatten the distances and
              voxel map to (num_voxels,).
        4. Remove the voxels that has -1 values in the voxel map which means that on this pixel was empty due to
              the projection or the label has been mapped to the ignore class.
        5. Add the model output with the voxel map and distances to the VoxelCloud object.
        6. Calculate the Viewpoint Entropy for each voxel in the VoxelCloud.
        7. Select the voxels with the highest Viewpoint Entropy and label them.

        :param dataset: Dataset object
        :param model: Model based on which the voxels will be selected
        :param percentage: Percentage of the dataset to be labeled (default: 1)
        """

        # Calculate, how many voxels should be labeled
        selection_size = self.get_selection_size(dataset, percentage)

        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        average_entropies = torch.tensor([], dtype=torch.float32)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for cloud in self.clouds:
                indices = np.where(dataset.cloud_map == cloud.path)[0]
                for i in tqdm(indices, desc='Mapping model output values to voxels'):
                    proj_image, proj_distances, proj_voxel_map, cloud_path = dataset.get_item(i)
                    proj_image = torch.from_numpy(proj_image).type(torch.float32).unsqueeze(0).to(self.device)
                    proj_distances = torch.from_numpy(proj_distances).type(torch.float32)
                    proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long)

                    # Forward pass
                    model_output = model(proj_image)

                    # Change the shape of the model output to (num_voxels, num_classes)
                    model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)
                    sample_distances = proj_distances.flatten()
                    sample_voxel_map = proj_voxel_map.flatten()

                    # Remove the voxels where voxel map is -1 (empty pixel or ignore class)
                    valid = (sample_voxel_map != -1)
                    model_output = model_output[valid]
                    sample_distances = sample_distances[valid]
                    sample_voxel_map = sample_voxel_map[valid]

                    cloud.add_predictions(model_output.cpu(), sample_distances, sample_voxel_map)

                entropies, cloud_voxel_map, cloud_cloud_map = cloud.get_average_entropies()
                average_entropies = torch.cat((average_entropies, entropies))
                voxel_map = torch.cat((voxel_map, cloud_voxel_map))
                cloud_map = torch.cat((cloud_map, cloud_cloud_map))

                cloud.reset()

            # Select the samples with the highest viewpoint entropy
            order = torch.argsort(average_entropies, descending=True)
            voxel_map, cloud_map = voxel_map[order], cloud_map[order]
            selected_voxels = voxel_map[:selection_size].cpu()
            selected_clouds = cloud_map[:selection_size].cpu()

            return self.get_voxel_selection(selected_voxels, selected_clouds)


class ViewpointVarianceVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_paths, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module, percentage: float = 1):

        # Calculate, how many voxels should be labeled
        selection_size = self.get_selection_size(dataset, percentage)

        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        viewpoint_variances = torch.tensor([], dtype=torch.float32)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for cloud in self.clouds:
                indices = np.where(dataset.cloud_map == cloud.path)[0]
                for i in tqdm(indices, desc='Mapping model output values to voxels'):
                    proj_image, _, proj_voxel_map, cloud_path = dataset.get_item(i)
                    proj_image = torch.from_numpy(proj_image).type(torch.float32).unsqueeze(0).to(self.device)
                    proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long)

                    # Forward pass
                    model_output = model(proj_image)

                    # Change the shape of the model output to (num_voxels, num_classes)
                    model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)
                    sample_voxel_map = proj_voxel_map.flatten()

                    # Remove the voxels where voxel map is -1 (empty pixel or ignore class)
                    valid = (sample_voxel_map != -1)
                    model_output = model_output[valid]
                    sample_voxel_map = sample_voxel_map[valid]

                    cloud.add_predictions(model_output.cpu(), sample_voxel_map, gradient=False, uncertainty=False)

                variances, cloud_voxel_map, cloud_cloud_map = cloud.get_viewpoint_variances()
                viewpoint_variances = torch.cat((viewpoint_variances, variances))
                voxel_map = torch.cat((voxel_map, cloud_voxel_map))
                cloud_map = torch.cat((cloud_map, cloud_cloud_map))

                cloud.reset()

            # Select the samples with the highest viewpoint entropy
            order = torch.argsort(viewpoint_variances, descending=True)
            voxel_map, cloud_map = voxel_map[order], cloud_map[order]
            selected_voxels = voxel_map[:selection_size].cpu()
            selected_clouds = cloud_map[:selection_size].cpu()

            return self.get_voxel_selection(selected_voxels, selected_clouds)


class EpistemicUncertaintyVoxelSelector(BaseVoxelSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, cloud_paths, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module, percentage: float = 1):

        # Calculate, how many voxels should be labeled
        selection_size = self.get_selection_size(dataset, percentage)

        voxel_map = torch.tensor([], dtype=torch.long)
        cloud_map = torch.tensor([], dtype=torch.long)
        epistemic_uncertainty = torch.tensor([], dtype=torch.float32)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for cloud in self.clouds:
                indices = np.where(dataset.cloud_map == cloud.path)[0]
                for i in tqdm(indices, desc='Mapping model output values to voxels'):
                    proj_image, _, proj_voxel_map, cloud_path = dataset.get_item(i)
                    proj_image = torch.from_numpy(proj_image).type(torch.float32).unsqueeze(0).to(self.device)
                    proj_voxel_map = torch.from_numpy(proj_voxel_map).type(torch.long)

                    sample_voxel_map = proj_voxel_map.flatten()
                    valid = (sample_voxel_map != -1)
                    sample_voxel_map = sample_voxel_map[valid]

                    model_output = torch.zeros((0,), dtype=torch.float32)

                    # Forward pass 10 times and concatenate the results
                    for j in range(10):
                        model_output_it = model(proj_image)
                        model_output_it = model_output_it.flatten(start_dim=2).permute(0, 2, 1)
                        model_output_it = model_output_it[:, valid, :]
                        model_output = torch.cat((model_output, model_output_it), dim=0)

                    cloud.add_predictions(model_output.cpu(), sample_voxel_map, mc_dropout=True)

                uncertainty, cloud_voxel_map, cloud_cloud_map = cloud.get_epistemic_uncertainty()
                epistemic_uncertainty = torch.cat((epistemic_uncertainty, uncertainty))
                voxel_map = torch.cat((voxel_map, cloud_voxel_map))
                cloud_map = torch.cat((cloud_map, cloud_cloud_map))

                cloud.reset()

            # Select the samples with the highest viewpoint entropy
            order = torch.argsort(epistemic_uncertainty, descending=True)
            voxel_map, cloud_map = voxel_map[order], cloud_map[order]
            selected_voxels = voxel_map[:selection_size].cpu()
            selected_clouds = cloud_map[:selection_size].cpu()

            return self.get_voxel_selection(selected_voxels, selected_clouds)
