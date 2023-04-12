import h5py
import torch
from torch.utils.data import Dataset


class SuperpointCloud(object):
    def __init__(self, path: str, size: int, superpoint_map: torch.Tensor, cloud_id: int):
        self.eps = 1e-6  # Small value to avoid division by zero
        self.path = path
        self.size = size
        self.superpoint_map = superpoint_map
        self.label_mask = torch.zeros((size,), dtype=torch.bool)

        self.id = cloud_id

        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.variances = torch.zeros((0,), dtype=torch.float32)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    @property
    def num_classes(self) -> int:
        """ Get the number of classes based on the given predictions.
        If no predictions are available, return None.
        """
        if self.predictions.shape != torch.Size([0]):
            return self.predictions.shape[1]
        else:
            return -1

    @property
    def num_superpoints(self) -> int:
        return self.superpoint_map.max().item() + 1

    def add_predictions(self, predictions: torch.Tensor, voxel_map: torch.Tensor, mc_dropout: bool = False) -> None:
        """ Add model predictions to the cloud.

        :param predictions: The model predictions to add to the cloud with shape (N, C), where N is the
                            number of inputs (pixels) and C is the number of classes.
        :param voxel_map: The voxel map with shape (N,), where N is the number of sample inputs (pixels).
                          The voxel map maps each prediction to a voxel in the cloud.
        :param mc_dropout: Whether the predictions are from a Monte Carlo dropout model.
        """

        if mc_dropout:
            variances = predictions.var(dim=0)
            predictions = predictions.mean(dim=0)
        else:
            variances = None

        # Remove the values of the voxels that are already labeled
        indices = torch.where(torch.isin(voxel_map, torch.nonzero(~self.label_mask).squeeze(1)))

        unlabeled_voxel_map = voxel_map[indices]
        self.voxel_map = torch.cat((self.voxel_map, unlabeled_voxel_map), dim=0)

        unlabeled_predictions = predictions[indices]
        self.predictions = torch.cat((self.predictions, unlabeled_predictions), dim=0)

        if variances is not None:
            unlabeled_variances = variances[indices]
            self.variances = torch.cat((self.variances, unlabeled_variances), dim=0)

    def reset(self) -> None:
        """ Reset the cloud by removing all predictions and other information. """

        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.variances = torch.zeros((0,), dtype=torch.float32)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    def label_voxels(self, voxels: torch.Tensor, dataset: Dataset = None) -> None:
        """ Label the given voxels in the dataset. The voxels are labeled by updating the `labeled_voxels` attribute,
        changing `label_map` of each sample on the disk and `selected_mask` of the sequence the cloud belongs to.

        :param voxels: The indices of the voxels to label
        :param dataset: The dataset object to label the voxels in
        """

        # Get indices where the voxels mask is True
        self.label_mask[voxels] = True
        if dataset is not None:
            with h5py.File(self.path, 'r+') as f:
                f['label_mask'][...] = self.label_mask.numpy()
            dataset.label_voxels(voxels.numpy(), self.path)

    def label_superpoints(self, superpoints: torch.Tensor, dataset: Dataset = None) -> None:
        voxels = torch.tensor([])
        for superpoint in superpoints:
            voxels = torch.cat((voxels, torch.nonzero(self.superpoint_map == superpoint).squeeze(1)), dim=0)
        self.label_voxels(voxels, dataset)

    def get_average_entropies(self):

        # Initialize the output
        average_entropies = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Split the stored values by voxel
        voxel_map, prediction_sets = self.cluster_by_voxels(self.voxel_map, self.predictions)

        # Calculate the average entropy for each voxel
        entropies = torch.tensor([])
        for prediction_set in prediction_sets:
            entropy = self.calculate_average_entropy(prediction_set).unsqueeze(0)
            entropies = torch.cat((entropies, entropy), dim=0)

        # Map the entropies to the voxels
        average_entropies[voxel_map] = entropies

        # Average the entropies by superpoint
        superpoint_average_entropies, superpoint_sizes = self.average_by_superpoint(average_entropies)
        return self._append_mapping(superpoint_average_entropies, superpoint_sizes)

    def get_viewpoint_variances(self):

        # Initialize the output
        viewpoint_variances = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Split the stored values by voxel
        voxel_map, prediction_sets = self.cluster_by_voxels(self.voxel_map, self.predictions)

        # Calculate the viewpoint variance for each voxel
        variances = torch.tensor([])
        for prediction_set in prediction_sets:
            var = self.calculate_variance(prediction_set).unsqueeze(0)
            var = var if not torch.isnan(var) else torch.zeros(1)
            variances = torch.cat((variances, var), dim=0)

        # Map the variances to the voxels
        viewpoint_variances[voxel_map] = variances

        # Average the variances by superpoint
        superpoint_viewpoint_variances, superpoint_sizes = self.average_by_superpoint(viewpoint_variances)
        return self._append_mapping(superpoint_viewpoint_variances, superpoint_sizes)

    def get_epistemic_uncertainties(self):

        # Initialize the output
        mean_variances = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Split the stored values by voxel
        voxel_map, _, variance_sets = self.cluster_by_voxels(self.voxel_map, self.predictions, self.variances)

        # Calculate the mean variance for each voxel
        variances = torch.tensor([])
        for variance_set in variance_sets:
            var = torch.mean(variance_set).unsqueeze(0)
            variances = torch.cat((variances, var), dim=0)

        # Map the variances to the voxels
        mean_variances[voxel_map] = variances

        # Average the variances by superpoint
        superpoint_mean_variances, superpoint_sizes = self.average_by_superpoint(mean_variances)
        return self._append_mapping(superpoint_mean_variances, superpoint_sizes)

    def average_by_superpoint(self, values: torch.Tensor):

        # Initialize the output
        average_superpoint_values = torch.full((self.num_superpoints,), float('nan'), dtype=torch.float32)
        superpoint_sizes = torch.zeros((self.num_superpoints,), dtype=torch.long)
        superpoints, sizes = torch.unique(self.superpoint_map, return_counts=True)
        superpoint_sizes[superpoints] = sizes

        # Average the values by superpoint
        for superpoint in torch.unique(self.superpoint_map):
            indices = torch.where(self.superpoint_map == superpoint)
            superpoint_values = values[indices]
            valid_superpoint_values = superpoint_values[~torch.isnan(superpoint_values)]
            if len(valid_superpoint_values) > 0:
                average_superpoint_values[superpoint] = torch.mean(valid_superpoint_values)
        return average_superpoint_values, superpoint_sizes

    @staticmethod
    def cluster_by_voxels(voxel_map: torch.Tensor, predictions: torch.Tensor, variances: torch.Tensor = None):
        """ Cluster the predictions and distances by voxel index.
        The shape of the predictions and distances is expected to by (N, C) and (N,) respectively, where N
        is the number of predictions and C is the number of classes. The shape of the voxel map is expected to be (N,).

        After clustering the predictions and distances are returned as a list of tensors, where each tensor
        contains the predictions or distances for a single voxel. The voxel map is returned as a tensor containing
        the unique voxel indices in ascending order.

        :param voxel_map: The voxel map of individual predictions
        :param predictions: The predictions to cluster
        :param variances: The variances to cluster
        :return: The unique voxel indices, the clustered predictions and distances
        """

        # Create an order for the predictions and distances based on the voxel map
        order = torch.argsort(voxel_map)
        unique_voxels, num_views = torch.unique(voxel_map, return_counts=True)

        # Split the predictions by voxel
        predictions = predictions[order]
        prediction_sets = torch.split(predictions, num_views.tolist())

        # Split the variances by voxel
        if variances is not None:
            distances = variances[order]
            distance_sets = torch.split(distances, num_views.tolist())
            return unique_voxels.type(torch.long), prediction_sets, distance_sets

        return unique_voxels.type(torch.long), prediction_sets

    def calculate_average_entropy(self, probability_distribution_set: torch.Tensor):
        entropies = self.calculate_entropy(probability_distribution_set)
        return torch.mean(entropies)

    def calculate_entropy(self, probability_distributions: torch.Tensor):
        probability_distributions = torch.clamp(probability_distributions, min=self.eps, max=1 - self.eps)
        entropies = torch.sum(- probability_distributions * torch.log(probability_distributions), dim=1)
        return entropies

    @staticmethod
    def calculate_variance(predictions: torch.Tensor) -> torch.Tensor:
        var = torch.var(predictions, dim=0)
        return torch.mean(var)

    def _append_mapping(self, superpoint_values: torch.Tensor, superpoint_sizes: torch.Tensor):
        """ Append the voxel indices and cloud ids to the values and return the result.

        :param superpoint_values: The values to append the mapping to
        :param superpoint_sizes: The sizes of the superpoints
        :return: A tuple containing the values, voxel indices and cloud ids
        """

        valid_indices = ~torch.isnan(superpoint_values)

        filtered_values = superpoint_values[valid_indices]
        filtered_superpoint_sizes = superpoint_sizes[valid_indices]
        filtered_superpoint_indices = torch.arange(self.num_superpoints, dtype=torch.long)[valid_indices]
        filtered_cloud_ids = torch.full((self.num_superpoints,), self.id, dtype=torch.long)[valid_indices]

        return filtered_values, filtered_superpoint_sizes, filtered_superpoint_indices, filtered_cloud_ids

    def __len__(self):
        return self.size

    def __str__(self):
        ret = f'\nVoxelCloud:\n' \
              f'\t - Cloud ID = {self.id}, \n' \
              f'\t - Cloud path = {self.path}, \n' \
              f'\t - Number of voxels in cloud = {self.size}\n' \
              f'\t - Number of superpoints in cloud = {self.num_superpoints}\n' \
              f'\t - Number of model predictions = {self.predictions.shape[0]}\n'
        if self.num_classes > 0:
            ret += f'\t - Number of semantic classes = {self.num_classes}\n'
        ret += f'\t - Number of already labeled voxels = {torch.sum(self.label_mask)}\n'
        return ret


if __name__ == '__main__':
    print('=========================================================')
    print('\tEXPERIMENT: 2 voxels, 5 predictions, 4 classes')
    print('=========================================================')

    # Simulate model outputs for semantic segmentation with 4 classes
    model_outputs_1 = torch.tensor([[8, 1, 4, 2],
                                    [4, 2, 3, 4],
                                    [9, 1, 3, 5],
                                    [1, 3, 2, 8],
                                    [3, 5, 2, 9],
                                    [3, 5, 2, 9]], dtype=torch.float32)

    model_outputs_2 = torch.tensor([[7, 2, 1, 2],
                                    [4, 2, 2, 3],
                                    [12, 1, 4, 5],
                                    [2, 2, 2, 9],
                                    [4, 1, 2, 8],
                                    [3, 5, 2, 9]], dtype=torch.float32)

    model_outputs_3 = torch.tensor([[8, 1, 3, 1],
                                    [4, 3, 1, 3],
                                    [10, 1, 3, 6],
                                    [2, 3, 1, 9],
                                    [3, 5, 2, 9],
                                    [3, 5, 2, 9]], dtype=torch.float32)

    # ------------------ WITHOUT MC DROPOUT ------------------

    model_outputs = torch.softmax(model_outputs_1, dim=1)
    voxel_mapping = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    superpoint_mapping = torch.tensor([0, 0, 1], dtype=torch.long)

    # Create a voxel cloud with 3 voxels
    cloud = SuperpointCloud(path='/path/to/cloud', size=3, superpoint_map=superpoint_mapping, cloud_id=0)

    print(f'\nCreating cloud with 3 voxels and adding 5 predictions:\n')
    cloud.add_predictions(model_outputs, voxel_mapping)
    print(cloud)

    calculated_viewpoint_entropies = cloud.get_average_entropies()
    print(f'\nViewpoint entropies:\n{calculated_viewpoint_entropies[0]}\n')

    calculated_viewpoint_variances = cloud.get_viewpoint_variance()
    print(f'\nViewpoint variances:\n{calculated_viewpoint_variances[0]}\n')

    # ------------------ WITH MC DROPOUT ------------------

    model_outputs_1 = torch.softmax(model_outputs_1, dim=1).unsqueeze(0)
    model_outputs_2 = torch.softmax(model_outputs_2, dim=1).unsqueeze(0)
    model_outputs_3 = torch.softmax(model_outputs_3, dim=1).unsqueeze(0)
    model_outputs = torch.cat((model_outputs_1, model_outputs_2, model_outputs_3), dim=0)
    voxel_mapping = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    superpoint_mapping = torch.tensor([0, 0, 1], dtype=torch.long)

    # Create a voxel cloud with 3 voxels
    cloud = SuperpointCloud(path='/path/to/cloud', size=3, superpoint_map=superpoint_mapping, cloud_id=0)

    cloud.add_predictions(model_outputs, voxel_mapping, mc_dropout=True)

    calculated_epistemic_uncertainties = cloud.get_epistemic_uncertainty()
    print(f'\nEpistemic uncertainties:\n{calculated_epistemic_uncertainties[0]}\n')
