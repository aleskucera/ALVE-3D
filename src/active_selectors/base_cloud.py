import h5py
import torch
from torch.utils.data import Dataset


class Cloud(object):
    def __init__(self, path: str, size: int, cloud_id: int):
        self.eps = 1e-6  # Small value to avoid division by zero
        self.path = path
        self.size = size
        self.id = cloud_id

        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.variances = torch.zeros((0,), dtype=torch.float32)
        self.label_mask = torch.zeros((size,), dtype=torch.bool)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    def _return_values(self, values: torch.Tensor) -> tuple:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        if self.predictions.shape != torch.Size([0]):
            return self.predictions.shape[1]
        else:
            return -1

    @property
    def percentage_labeled(self) -> float:
        return torch.sum(self.label_mask).item() / self.size

    @property
    def selection_key(self) -> str:
        split = self.path.split('/')
        key = '/'.join(split[-3:])
        return key

    def add_predictions(self, predictions: torch.Tensor, voxel_map: torch.Tensor, mc_dropout: bool = False) -> None:
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
        self.voxel_map = torch.zeros((0,), dtype=torch.int32)
        self.variances = torch.zeros((0,), dtype=torch.float32)
        self.predictions = torch.zeros((0,), dtype=torch.float32)

    def label_voxels(self, voxels: torch.Tensor, dataset: Dataset = None) -> None:
        self.label_mask[voxels] = True
        if dataset is not None:
            with h5py.File(self.path.replace('sequences', dataset.project_name), 'w') as f:
                f.create_dataset('label_mask', data=self.label_mask.numpy())
            dataset.label_voxels(voxels.numpy(), self.path)

    def get_average_entropies(self) -> tuple:
        return self.__get_values(self.predictions, self.__average_entropy)

    def get_viewpoint_entropies(self) -> tuple:
        return self.__get_values(self.predictions, self.__viewpoint_entropy)

    def get_viewpoint_variances(self) -> tuple:
        return self.__get_values(self.predictions, self.__variance)

    def get_epistemic_uncertainties(self) -> tuple:
        return self.__get_values(self.variances, torch.mean)

    def __get_values(self, items: torch.Tensor, function: callable):
        values = torch.full((self.size,), float('nan'), dtype=torch.float32)

        order = torch.argsort(self.voxel_map)
        unique_voxels, num_views = torch.unique(self.voxel_map, return_counts=True)

        items = items[order]
        voxel_map = unique_voxels.type(torch.long)
        item_sets = torch.split(items, num_views.tolist())

        vals = torch.tensor([])
        for item_set in item_sets:
            val = function(item_set).unsqueeze(0)
            vals = torch.cat((vals, val), dim=0)

        values[voxel_map] = vals
        return self._return_values(values)

    def __average_entropy(self, probability_distribution_set: torch.Tensor) -> torch.Tensor:
        probability_distribution_set = torch.clamp(probability_distribution_set, min=self.eps, max=1 - self.eps)
        entropies = torch.sum(- probability_distribution_set * torch.log(probability_distribution_set), dim=1)
        return torch.mean(entropies)

    def __viewpoint_entropy(self, probability_distribution_set: torch.Tensor) -> torch.Tensor:
        mean_distribution = torch.mean(probability_distribution_set, dim=0)
        mean_distribution = torch.clamp(mean_distribution, min=self.eps, max=1 - self.eps)
        return torch.sum(- mean_distribution * torch.log(mean_distribution))

    @staticmethod
    def __variance(predictions: torch.Tensor) -> torch.Tensor:
        var = torch.var(predictions, dim=0)
        return torch.mean(var)

    @staticmethod
    def __cluster_by_voxels(items: torch.Tensor, voxel_map: torch.Tensor) -> tuple:
        order = torch.argsort(voxel_map)
        unique_voxels, num_views = torch.unique(voxel_map, return_counts=True)

        items = items[order]
        item_sets = torch.split(items, num_views.tolist())
        return item_sets, unique_voxels.type(torch.long)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return self.__str__()
