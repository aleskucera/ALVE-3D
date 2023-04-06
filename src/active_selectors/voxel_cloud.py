import h5py
import torch
from torch.utils.data import Dataset


class VoxelCloud(object):
    """ An object that represents a global cloud in a dataset sequence. The voxels are areas of space that can contain
    multiple points from different frames. The cloud is a collection of these voxels.
    This class is used to store model predictions and other information for each voxel in the cloud. These predictions
    can be used to select the most informative voxels for active learning.

    We can calculate following metrics for each voxel:
     - Viewpoint Entropy: The entropy of the model predictions for the voxel from all viewpoints.
     - Weighted Viewpoint Entropy: The entropy of the model predictions for the voxel from all viewpoints weighted by
                                   the distance of the viewpoint to the voxel.
     - Viewpoint Cross Entropy: The cross entropy of the model predictions and distances for
                                the voxel from all viewpoints.

    :param size: The number of voxels in the cloud
    :param cloud_id: The id of the cloud (used to identify the cloud in the dataset, unique for each cloud in dataset)
    """

    def __init__(self, path: str, size: int, label_mask: torch.Tensor, cloud_id: int):
        self.eps = 1e-6  # Small value to avoid division by zero
        self.path = path
        self.size = size
        self.label_mask = label_mask

        self.id = cloud_id

        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.variances = torch.zeros((0,), dtype=torch.float32)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    @property
    def num_classes(self) -> int:
        """ Get the number of classes based on the given predictions.
        If no predictions are available, return None.
        """
        if len(self.predictions.shape) > 0:
            return self.predictions.shape[1]
        else:
            return -1

    @staticmethod
    def percentage_labeled(self) -> float:
        """ Get the percentage of voxels that are labeled in the cloud. """
        return torch.sum(self.label_mask).item() / self.size

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
            with h5py.File(self.path.replace('sequences', dataset.project_name), 'w') as f:
                f.create_dataset('label_mask', data=self.label_mask.numpy())
            dataset.label_voxels(voxels.numpy(), self.path)

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

        average_entropies[voxel_map] = entropies
        return self._append_mapping(average_entropies)

    def get_viewpoint_variance(self):

        # Initialize the output
        viewpoint_variances = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Split the stored values by voxel
        voxel_map, prediction_sets = self.cluster_by_voxels(self.voxel_map, self.predictions)

        # Calculate the viewpoint variance for each voxel
        variances = torch.tensor([])
        for prediction_set in prediction_sets:
            var = self.calculate_variance(prediction_set).unsqueeze(0)
            variances = torch.cat((variances, var), dim=0)

        viewpoint_variances[voxel_map] = variances
        return self._append_mapping(viewpoint_variances)

    def get_epistemic_uncertainty(self):

        # Initialize the output
        mean_variances = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Split the stored values by voxel
        voxel_map, _, variance_sets = self.cluster_by_voxels(self.voxel_map, self.predictions, self.variances)

        # Calculate the mean variance for each voxel
        variances = torch.tensor([])
        for variance_set in variance_sets:
            var = torch.mean(variance_set).unsqueeze(0)
            variances = torch.cat((variances, var), dim=0)

        mean_variances[voxel_map] = variances
        return self._append_mapping(mean_variances)

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

    def _append_mapping(self, values: torch.Tensor):
        """ Append the voxel indices and cloud ids to the values and return the result.

        :param values: The values to append the mapping to
        :return: A tuple containing the values, voxel indices and cloud ids
        """

        valid_indices = ~torch.isnan(values)

        filtered_values = values[valid_indices]
        filtered_voxel_indices = torch.arange(self.size, dtype=torch.long)[valid_indices]
        filtered_cloud_ids = torch.full((self.size,), self.id, dtype=torch.long)[valid_indices]

        return filtered_values, filtered_voxel_indices, filtered_cloud_ids

    def __len__(self):
        return self.size

    def __str__(self):
        ret = f'\nVoxelCloud:\n' \
              f'\t - Cloud ID = {self.id}, \n' \
              f'\t - Cloud path = {self.path}, \n' \
              f'\t - Number of voxels in cloud = {self.size}\n' \
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
                                    [3, 5, 2, 9]], dtype=torch.float32)

    model_outputs_2 = torch.tensor([[8, 1, 4, 2],
                                    [4, 2, 3, 4],
                                    [9, 1, 3, 5],
                                    [1, 3, 2, 8],
                                    [3, 5, 2, 9]], dtype=torch.float32)

    model_outputs_3 = torch.tensor([[8, 1, 4, 2],
                                    [4, 2, 3, 4],
                                    [9, 1, 3, 5],
                                    [1, 3, 2, 8],
                                    [3, 5, 2, 9]], dtype=torch.float32)

    # ------------------ WITHOUT MC DROPOUT ------------------

    model_outputs = torch.softmax(model_outputs_1, dim=1)
    voxel_mapping = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    # Create a voxel cloud with 3 voxels
    cloud = VoxelCloud(path='/path/to/cloud', size=3, label_mask=torch.zeros(3, dtype=torch.bool), cloud_id=0)

    print(f'\nCreating cloud with 3 voxels and adding 5 predictions:\n')
    cloud.add_predictions(model_outputs, voxel_mapping)

    calculated_viewpoint_entropies = cloud.get_average_entropies()
    print(f'\nViewpoint entropies:\n{calculated_viewpoint_entropies[0]}\n')

    calculated_viewpoint_variances = cloud.get_viewpoint_variance()
    print(f'\nViewpoint variances:\n{calculated_viewpoint_variances[0]}\n')

    # ------------------ WITH MC DROPOUT ------------------

    model_outputs_1 = torch.softmax(model_outputs_1, dim=1).unsqueeze(0)
    model_outputs_2 = torch.softmax(model_outputs_2, dim=1).unsqueeze(0)
    model_outputs_3 = torch.softmax(model_outputs_3, dim=1).unsqueeze(0)
    model_outputs = torch.cat((model_outputs_1, model_outputs_2, model_outputs_3), dim=0)
    voxel_mapping = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    # Create a voxel cloud with 3 voxels
    cloud = VoxelCloud(path='/path/to/cloud', size=3, label_mask=torch.zeros(3, dtype=torch.bool), cloud_id=0)

    cloud.add_predictions(model_outputs, voxel_mapping, mc_dropout=True)

    calculated_epistemic_uncertainties = cloud.get_epistemic_uncertainty()
    print(f'\nEpistemic uncertainties:\n{calculated_epistemic_uncertainties[0]}\n')
