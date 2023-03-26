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
        # self.sequence = sequence
        # self.seq_cloud_id = seq_cloud_id

        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.distances = torch.zeros((0,), dtype=torch.float32)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    @property
    def num_classes(self) -> int:
        """ Get the number of classes based on the given predictions.
        If no predictions are available, return None.
        """

        return self.predictions.shape[1] if len(self.predictions.shape) > 0 else -1

    def add_predictions(self, predictions: torch.Tensor, distances: torch.Tensor, voxel_map: torch.Tensor) -> None:
        """ Add model predictions to the cloud.

        :param predictions: The model predictions to add to the cloud with shape (N, C), where N is the
                            number of inputs (pixels) and C is the number of classes.
        :param distances: The radial distances of the points for which the prediction has been made with shape (N,).
        :param voxel_map: The voxel map with shape (N,), where N is the number of sample inputs (pixels).
                          The voxel map maps each prediction to a voxel in the cloud.
        """

        # Remove the values of the voxels that are already labeled
        indices = torch.where(torch.isin(voxel_map, torch.nonzero(~self.label_mask).squeeze(1)))
        unlabeled_predictions = predictions[indices]
        unlabeled_distances = distances[indices]
        unlabeled_voxel_map = voxel_map[indices]

        # Append new values to existing values
        self.predictions = torch.cat((self.predictions, unlabeled_predictions), dim=0)
        self.distances = torch.cat((self.distances, unlabeled_distances), dim=0)
        self.voxel_map = torch.cat((self.voxel_map, unlabeled_voxel_map), dim=0)

    def reset(self) -> None:
        """ Reset the cloud by removing all predictions and other information. """

        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.distances = torch.zeros((0,), dtype=torch.float32)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    def label_voxels(self, voxels: torch.Tensor, dataset: Dataset = None) -> None:
        """ Label the given voxels in the dataset. The voxels are labeled by updating the `labeled_voxels` attribute,
        changing `label_map` of each sample on the disk and `selected_mask` of the sequence the cloud belongs to.

        :param voxels: The indices of the voxels to label
        :param dataset: The dataset object to label the voxels in
        """

        # voxels = voxels.cpu()

        # Get indices where the voxels mask is True
        self.label_mask[voxels] = True
        if dataset is not None:
            with h5py.File(self.path, 'r+') as f:
                f['label_mask'][...] = self.label_mask.numpy()
            dataset.label_voxels(voxels.numpy(), self.path)

    def get_viewpoint_entropies(self):
        """ Calculate the viewpoint entropy for each voxel in the cloud. This is done by executing the following steps:

        1. Cluster all predictions with the same voxel index together into a set.
        2. Calculate the mean of the predictions for each class. This results into a single probability distribution
           for each voxel instead of set of distributions.
        3. Calculate the entropy of this probability distribution for each voxel.
        4. Append the mapping to the entropies and return the result.
        """

        # Initialize the output
        viewpoint_entropies = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Split the stored values by voxel
        voxel_map, prediction_sets = self.cluster_by_voxels(self.voxel_map, self.predictions)

        # Calculate the entropy for each voxel from mean of all viewpoints
        entropies = torch.tensor([])
        for prediction_set in prediction_sets:
            entropy = self.calculate_mean_entropy(prediction_set).unsqueeze(0)
            entropies = torch.cat((entropies, entropy), dim=0)

        viewpoint_entropies[voxel_map] = entropies
        return self._append_mapping(viewpoint_entropies)

    def get_weighted_viewpoint_entropies(self):
        """ Calculate the weighted viewpoint entropy for each voxel in the cloud.
        This is done by executing the following steps:

        1. Cluster all predictions and distances with the same voxel index together into a set.
        2. Calculate the weighted mean of the predictions for each class. The weights are the inverted
           values of the distances. This results into a single probability distribution for each voxel
           instead of set of distributions.
        3. Calculate the entropy of this probability distribution for each voxel.
        4. Append the mapping to the entropies and return the result.
        """

        # Initialize the output
        weighted_viewpoint_entropies = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Split the stored values by voxel
        voxel_map, prediction_sets, distance_sets = self.cluster_by_voxels(self.voxel_map, self.predictions,
                                                                           self.distances)

        # Calculate the weighted viewpoint entropy for each voxel
        entropies = torch.tensor([])
        for prediction_set, distance_set in zip(prediction_sets, distance_sets):
            entropy = self.calculate_weighted_entropy(prediction_set, distance_set).unsqueeze(0)
            entropies = torch.cat((entropies, entropy), dim=0)

        weighted_viewpoint_entropies[voxel_map] = entropies
        return self._append_mapping(weighted_viewpoint_entropies)

    def get_cross_entropies(self):
        """ Calculate the cross entropy for each voxel in the cloud. This is done by executing the following steps:

        1. Calculate the entropy of each prediction.
        2. Cluster all entropies and distances with the same voxel index together into a set.
        3. Create a probability distribution from the distances and entropies for each voxel by normalizing them.
        4. Calculate the cross entropy between the entropy of the predictions and the distance for each voxel.
        5. Append the mapping to the entropies and return the result.
        """

        # Initialize the output
        cross_entropies = torch.full((self.size,), float('nan'), dtype=torch.float32)

        # Calculate the entropy of the predictions
        pred_entropies = self.calculate_entropy(self.predictions)

        # Split the values by voxel
        voxel_map, entropy_sets, distance_sets = self.cluster_by_voxels(self.voxel_map, pred_entropies, self.distances)

        # Calculate the cross entropy for each voxel between the entropy of the predictions and the distance
        entropies = torch.tensor([])
        for entropy_set, distance_set in zip(entropy_sets, distance_sets):
            entropy = self.calculate_cross_entropy(entropy_set, distance_set).unsqueeze(0)
            entropies = torch.cat((entropies, entropy), dim=0)

        cross_entropies[voxel_map] = entropies
        return self._append_mapping(cross_entropies)

    @staticmethod
    def cluster_by_voxels(voxel_map: torch.Tensor, predictions: torch.Tensor, distances: torch.Tensor = None):
        """ Cluster the predictions and distances by voxel index.
        The shape of the predictions and distances is expected to by (N, C) and (N,) respectively, where N
        is the number of predictions and C is the number of classes. The shape of the voxel map is expected to be (N,).

        After clustering the predictions and distances are returned as a list of tensors, where each tensor
        contains the predictions or distances for a single voxel. The voxel map is returned as a tensor containing
        the unique voxel indices in ascending order.

        :param voxel_map: The voxel map of individual predictions
        :param predictions: The predictions to cluster
        :param distances: The distances to cluster
        :return: The unique voxel indices, the clustered predictions and distances
        """

        # Create an order for the predictions and distances based on the voxel map
        order = torch.argsort(voxel_map)
        unique_voxels, num_views = torch.unique(voxel_map, return_counts=True)

        # Split the predictions by voxel
        predictions = predictions[order]
        prediction_sets = torch.split(predictions, num_views.tolist())

        # Split the distances by voxel
        if distances is not None:
            distances = distances[order]
            distance_sets = torch.split(distances, num_views.tolist())
            return unique_voxels.type(torch.long), prediction_sets, distance_sets

        return unique_voxels.type(torch.long), prediction_sets

    def calculate_mean_entropy(self, probability_distribution_set: torch.Tensor):
        mean_distribution = torch.mean(probability_distribution_set, dim=0)
        mean_distribution = torch.clamp(mean_distribution, min=self.eps, max=1 - self.eps)
        entropy = torch.sum(- mean_distribution * torch.log(mean_distribution))
        return entropy

    def calculate_weighted_entropy(self, probability_distribution_set: torch.Tensor, distance_set: torch.Tensor):
        distance_weights = 1 / (distance_set * torch.sum(distance_set))
        weighted_distribution = torch.sum(probability_distribution_set * distance_weights.unsqueeze(1), dim=0)
        weighted_distribution = torch.clamp(weighted_distribution, min=self.eps, max=1 - self.eps)
        entropy = torch.sum(- weighted_distribution * torch.log(weighted_distribution))
        return entropy

    def calculate_cross_entropy(self, entropy_set: torch.Tensor, distance_set: torch.Tensor):
        entropy_distribution = entropy_set / torch.sum(entropy_set)
        entropy_distribution = torch.clamp(entropy_distribution, min=self.eps, max=1 - self.eps)
        distance_distribution = 1 / (distance_set * torch.sum(distance_set))
        distance_distribution = torch.clamp(distance_distribution, min=self.eps, max=1 - self.eps)
        cross_entropy = torch.sum(-distance_distribution * torch.log(entropy_distribution))
        return cross_entropy

    def calculate_entropy(self, probability_distributions: torch.Tensor):
        probability_distributions = torch.clamp(probability_distributions, min=self.eps, max=1 - self.eps)
        entropies = torch.sum(- probability_distributions * torch.log(probability_distributions), dim=1)
        return entropies

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
    model_outputs = torch.tensor([[8, 1, 4, 2],
                                  [4, 2, 3, 4],
                                  [9, 1, 3, 5],
                                  [1, 3, 2, 8],
                                  [3, 5, 2, 9]], dtype=torch.float32)
    print(f'\nModel raw outputs:\n{model_outputs}\n')

    # Softmax the outputs to get probabilities
    model_outputs = torch.softmax(model_outputs, dim=1)
    print(f'\nModel softmax outputs:\n{model_outputs}\n')

    # Corresponding radial distances for each prediction
    radial_distances = torch.tensor([1, 3, 2, 0.5, 2], dtype=torch.float32)
    print(f'\nRadial distances:\n{radial_distances}\n')

    # This is the voxel map that maps each prediction to a voxel in the cloud
    voxel_mapping = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    print(f'\nVoxel map:\n{voxel_mapping}\n')

    # Create a voxel cloud with 3 voxels
    cloud = VoxelCloud(path='/path/to/cloud', size=3, label_mask=torch.zeros(3, dtype=torch.bool), cloud_id=0)

    print(f'\nCreating cloud with 3 voxels and adding 5 predictions:\n')
    cloud.add_predictions(model_outputs, radial_distances, voxel_mapping)
    print(cloud)

    calculated_viewpoint_entropies = cloud.get_viewpoint_entropies()
    print(f'\nViewpoint entropies:\n{calculated_viewpoint_entropies[0]}\n')

    calculated_weighted_viewpoint_entropies = cloud.get_weighted_viewpoint_entropies()
    print(f'\nWeighted viewpoint entropies:\n{calculated_weighted_viewpoint_entropies[0]}\n')

    calculated_cross_entropies = cloud.get_cross_entropies()
    print(f'\nCross entropies:\n{calculated_cross_entropies[0]}\n')
