import torch
import logging
from torch_scatter import scatter_mean, scatter_std

from src.datasets import Dataset

log = logging.getLogger(__name__)


class Cloud(object):
    """ Base class for clouds. The clouds are used to select voxels from the dataset.
    The object provides following functionalities:
        - Add predictions of the model and map them to the voxels
        - Label selected voxels
        - Calculate the following metrics for each voxel:
            - Average entropy
            - Viewpoint entropy
            - Viewpoint variance
            - Epistemic uncertainty

    After calculating the metrics for the voxels, the calculated values are saved in the
    cloud and all other information is deleted. This is done to save memory.

    :param path: Path to the cloud
    :param size: Number of voxels in the cloud
    :param cloud_id: Unique id of the cloud
    """

    def __init__(self, path: str, size: int, cloud_id: int,
                 diversity_aware: bool, labels: torch.Tensor,
                 surface_variation: torch.Tensor,
                 color_discontinuity: torch.Tensor = None):

        self.eps = 1e-6  # Small value to avoid division by zero
        self.path = path
        self.size = size
        self.id = cloud_id
        self.labels = labels

        self.diversity_aware = diversity_aware
        self.surface_variation = surface_variation
        self.color_discontinuity = color_discontinuity if color_discontinuity is not None \
            else torch.zeros_like(self.surface_variation)

        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.variances = torch.zeros((0,), dtype=torch.float32)
        self.label_mask = torch.zeros((size,), dtype=torch.bool)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    def _save_metric(self, values: torch.Tensor, features: torch.Tensor = None) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """ Returns the number of semantic classes in the cloud. If the cloud does not contain predictions
        for semantic classes, -1 is returned.
        """

        if self.predictions.shape != torch.Size([0]):
            return self.predictions.shape[1]
        else:
            return -1

    @property
    def percentage_labeled(self) -> float:
        """ Returns the percentage of labeled voxels in the cloud.
        """

        return torch.sum(self.label_mask).item() / self.size

    @property
    def selection_key(self) -> str:
        """ Returns the key that is used to identify the cloud in the selection dictionary. Each cloud has a
        unique path, but the path can change with different devices. Therefore, the key is created by taking the
        last three parts of the path and joining them with a slash.

        Example:
            self.path = /home/user/dataset/sequences/03/voxel_clouds/000131_000634.h5
            key = 03/voxel_clouds/000131_000634.h5
        """

        split = self.path.split('/')
        key = '/'.join(split[-3:])
        return key

    @property
    def mean_predictions(self) -> torch.Tensor:
        mean_predictions = scatter_mean(self.predictions, self.voxel_map, dim=0, dim_size=self.size)
        zero_fill = torch.zeros((self.size - mean_predictions.shape[0], mean_predictions.shape[1]))
        voxel_mean_predictions = torch.cat((mean_predictions, zero_fill))
        return voxel_mean_predictions

    @property
    def mean_variances(self) -> torch.Tensor:
        mean_variances = scatter_mean(self.variances, self.voxel_map, dim=0, dim_size=self.size)
        zero_fill = torch.zeros((self.size - mean_variances.shape[0],))
        voxel_mean_variances = torch.cat((mean_variances, zero_fill))
        return voxel_mean_variances

    @property
    def std_predictions(self) -> torch.Tensor:
        std_predictions = scatter_std(self.predictions, self.voxel_map, dim=0)
        zero_fill = torch.zeros((self.size - std_predictions.shape[0], std_predictions.shape[1]))
        voxel_std_predictions = torch.cat((std_predictions, zero_fill))
        return voxel_std_predictions

    def add_predictions(self, predictions: torch.Tensor, voxel_map: torch.Tensor, mc_dropout: bool = False) -> None:
        """ Adds the predictions of the model to the cloud. The predictions are mapped to the voxels and the
        predictions of the voxels that are already labeled are removed.

        :param predictions: Predictions of the model. The expected shape is (N, C) where N is the number of voxels
                            and C is the number of semantic classes. The other option is (M, N, C) where M is the
                            number of predictions (e.g. for MC dropout) and N and C are the same as before. In this
                            case, the information stored for future calculations is the mean of the predictions and
                            the variance. (must be specified with mc_dropout)
        :param voxel_map: Mapping of the voxels to the point cloud. The expected shape is (N,) where N is the number
                            of voxels. The values of the tensor are the indices of the points in the point cloud.
        :param mc_dropout: If True, the predictions are assumed to be the result of MC dropout and the mean and
                            variance are calculated.
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

    def label_voxels(self, voxels: torch.Tensor, dataset: Dataset = None) -> None:
        """ Labels the voxels in the cloud.

        :param voxels: Indices of the voxels that should be labeled.
        :param dataset: Dataset that contains the point cloud. If specified, the voxels are also labeled in the dataset and
                written to the disk.
        """

        self.label_mask[voxels] = True
        if dataset is not None:
            dataset.label_voxels(voxels.numpy(), self.path)

    def compute_viewpoint_variance(self) -> None:
        viewpoint_deviations = scatter_std(self.predictions, self.voxel_map, dim=0, dim_size=self.size).mean(dim=1)
        voxel_mean_predictions = scatter_mean(self.predictions, self.voxel_map, dim=0, dim_size=self.size)
        features = voxel_mean_predictions if self.diversity_aware else None
        self._save_metric(viewpoint_deviations, features=features)
        self.__reset()

    def compute_epistemic_uncertainty(self) -> None:
        epistemic_uncertainties = scatter_mean(self.variances, self.voxel_map, dim=0, dim_size=self.size).mean(dim=1)
        voxel_mean_predictions = scatter_mean(self.predictions, self.voxel_map, dim=0, dim_size=self.size)
        features = voxel_mean_predictions if self.diversity_aware else None
        self._save_metric(epistemic_uncertainties, features=features)
        self.__reset()

    def compute_redal_score(self, weights: list[float] = None) -> None:
        voxel_mean_predictions = scatter_mean(self.predictions, self.voxel_map, dim=0, dim_size=self.size)
        features = voxel_mean_predictions if self.diversity_aware else None
        voxel_mean_predictions = torch.clamp(voxel_mean_predictions, min=self.eps, max=1 - self.eps)
        entropy = -torch.sum(voxel_mean_predictions * torch.log(voxel_mean_predictions), dim=1)
        redal_score = weights[0] * entropy + \
                      weights[1] * self.color_discontinuity + \
                      weights[2] * self.surface_variation
        self._save_metric(redal_score, features=features)
        self.__reset()

    def compute_entropy(self) -> None:
        voxel_mean_predictions = scatter_mean(self.predictions, self.voxel_map, dim=0, dim_size=self.size)
        features = voxel_mean_predictions if self.diversity_aware else None
        voxel_mean_predictions = torch.clamp(voxel_mean_predictions, min=self.eps, max=1 - self.eps)
        entropy = -torch.sum(voxel_mean_predictions * torch.log(voxel_mean_predictions), dim=1)
        self._save_metric(entropy, features=features)
        self.__reset()

    def compute_margin(self) -> None:
        voxel_mean_predictions = scatter_mean(self.predictions, self.voxel_map, dim=0, dim_size=self.size)
        sorted_predictions = torch.sort(voxel_mean_predictions, dim=1, descending=True)[0]
        margin = sorted_predictions[:, 0] - sorted_predictions[:, 1]
        features = voxel_mean_predictions if self.diversity_aware else None
        self._save_metric(margin, features=features)
        self.__reset()

    def compute_confidence(self) -> None:
        voxel_mean_predictions = scatter_mean(self.predictions, self.voxel_map, dim=0, dim_size=self.size)
        voxel_confidence = torch.max(voxel_mean_predictions, dim=1)[0]
        least_confidence = 1 - voxel_confidence
        features = voxel_mean_predictions if self.diversity_aware else None
        self._save_metric(least_confidence, features=features)
        self.__reset()

    # def __calculate_metric(self, values: torch.Tensor, function: callable) -> None:
    #     """ Calculates the metric for the voxels in the cloud specified by the function argument.
    #
    #     :param values: Items that should be used to calculate the metric.
    #                   The expected shape is (N, C) where N is the number
    #     :param function: Function that is used to calculate the metric. The function must take a tensor
    #                      of shape (N, C) and return a scalar value.
    #     """
    #     log.info('Calculating metric for cloud: ' + self.path)
    #     start = time.time()
    #     metric = torch.full((self.size,), float('nan'), dtype=torch.float32)
    #     features = torch.full((self.size, self.num_classes), float('nan'),
    #                           dtype=torch.float32) if self.diversity_aware else None
    #
    #     order = torch.argsort(self.voxel_map)
    #     unique_voxels, num_views = torch.unique(self.voxel_map, return_counts=True)
    #
    #     values = values[order]
    #     voxel_map = unique_voxels.type(torch.long)
    #     value_sets = torch.split(values, num_views.tolist())
    #
    #     vals = torch.tensor([])
    #     feats = torch.tensor([])
    #     for value_set in value_sets:
    #         vals = torch.cat((vals, function(value_set).unsqueeze(0)), dim=0)
    #         if self.diversity_aware:
    #             feats = torch.cat((feats, value_set.mean(dim=0).unsqueeze(0)), dim=0)
    #
    #     log.info(f'Calculating metric for {self.path} took {time.time() - start} seconds.')
    #     start = time.time()
    #     metric[voxel_map] = vals
    #     if self.diversity_aware:
    #         features[voxel_map] = feats
    #         self._save_metric(metric, features)
    #     else:
    #         self._save_metric(metric)
    #     log.info(f'Saving metric for {self.path} took {time.time() - start} seconds.')

    def __reset(self) -> None:
        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.variances = torch.zeros((0,), dtype=torch.float32)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    # def __average_entropy(self, probability_distribution_set: torch.Tensor) -> torch.Tensor:
    #     probability_distribution_set = torch.clamp(probability_distribution_set, min=self.eps, max=1 - self.eps)
    #     entropies = torch.sum(- probability_distribution_set * torch.log(probability_distribution_set), dim=1)
    #     return torch.mean(entropies)
    #
    # def __viewpoint_entropy(self, probability_distribution_set: torch.Tensor) -> torch.Tensor:
    #     mean_distribution = torch.mean(probability_distribution_set, dim=0)
    #     mean_distribution = torch.clamp(mean_distribution, min=self.eps, max=1 - self.eps)
    #     return torch.sum(- mean_distribution * torch.log(mean_distribution))
    #
    # @staticmethod
    # def __least_confident(probability_distribution_set: torch.Tensor) -> torch.Tensor:
    #     max_probabilities = torch.max(probability_distribution_set, dim=1)[0]
    #     return torch.mean(1 - max_probabilities)
    #
    # @staticmethod
    # def __margin(probability_distribution_set: torch.Tensor) -> torch.Tensor:
    #     sorted_probabilities = torch.sort(probability_distribution_set, dim=1)[0]
    #     margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    #     return torch.mean(margin)
    #
    # @staticmethod
    # def __variance(predictions: torch.Tensor) -> torch.Tensor:
    #     var = torch.var(predictions, dim=0)
    #     return torch.mean(var)

    # @staticmethod
    # def __cluster_by_voxels(items: torch.Tensor, voxel_map: torch.Tensor) -> tuple:
    #     order = torch.argsort(voxel_map)
    #     unique_voxels, num_views = torch.unique(voxel_map, return_counts=True)
    #
    #     items = items[order]
    #     item_sets = torch.split(items, num_views.tolist())
    #     return item_sets, unique_voxels.type(torch.long)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return self.__str__()
