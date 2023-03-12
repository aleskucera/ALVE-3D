import torch
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

    def add_values(self, predictions: torch.Tensor, voxel_map: torch.Tensor) -> None:
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


if __name__ == '__main__':
    model_outputs = torch.tensor([[8, 1, 4, 2], [4, 2, 3, 4], [9, 1, 3, 5], [1, 3, 2, 8], [3, 5, 2, 9]],
                                 dtype=torch.float32)
    model_probs = torch.softmax(model_outputs, dim=1)
    voxel_map = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    cloud = VoxelCloud(3, 0, 0, 0, torch.device('cpu'))

    cloud.add_values(model_probs, voxel_map)
    print(cloud)
    mean_values = cloud.get_viewpoint_entropies()
    print(mean_values)
