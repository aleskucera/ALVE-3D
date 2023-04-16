import h5py
import torch
import wandb
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .cloud import SuperpointCloud
from src.laserscan import LaserScan


class BaseSuperpointSelector:
    """ Base class for voxel selectors

    :param dataset_path: Path to the dataset
    :param cloud_paths: List of paths to the sequence clouds
    :param device: Device to use for computations
    """

    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device):
        self.device = device
        self.dataset_path = dataset_path

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
                superpoint_map = torch.tensor(f['superpoints'][:], dtype=torch.long)
                self.num_voxels += num_voxels
                self.clouds.append(SuperpointCloud(cloud_path, num_voxels, cloud_id, superpoint_map))

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

    def get_voxel_selection(self, selected_superpoints: torch.Tensor, cloud_map: torch.Tensor):
        voxel_selection = dict()
        for cloud in self.clouds:
            superpoints = selected_superpoints[cloud_map == cloud.id]
            for superpoint in superpoints:
                voxels = torch.nonzero(cloud.superpoint_map == superpoint).squeeze(1)
                cloud.label_voxels(voxels)
            split = cloud.path.split('/')
            name = '/'.join(split[-3:])
            voxel_selection[name] = cloud.label_mask
        return voxel_selection

    def load_voxel_selection(self, voxel_selection: dict, dataset: Dataset = None):
        for cloud_name, label_mask in voxel_selection.items():
            cloud = self.get_cloud(cloud_name)
            voxels = torch.nonzero(label_mask).squeeze(1)
            cloud.label_voxels(voxels, dataset)

    @staticmethod
    def log_selection(cfg: DictConfig, dataset: Dataset, save: bool = False) -> tuple[np.ndarray, float]:

        # --------------------------------------------------------
        # ================== DATASET STATISTICS ==================
        # --------------------------------------------------------

        ignore_index = cfg.ds.ignore_index
        label_names = [v for k, v in cfg.ds.labels_train.items() if k != ignore_index]
        class_dist, labeled_class_distribution, class_labeling_progress, labeled_ratio = dataset.get_statistics()

        # Log the dataset labeling progress
        wandb.log({f'Dataset Labeling Progress': labeled_ratio}, step=0)

        # Filter and log the dataset class distribution
        class_dist = np.delete(class_dist, ignore_index)
        data = [[name, value] for name, value in zip(label_names, class_dist)]
        table = wandb.Table(data=data, columns=["Class", "Distribution"])
        wandb.log({f"Class Distribution - "
                   f"{labeled_ratio:.2f}%": wandb.plot.bar(table, "Class", "Distribution")}, step=0)

        # Log the labeled class distribution
        labeled_class_dist = np.delete(labeled_class_distribution, ignore_index)
        data = [[name, value] for name, value in zip(label_names, labeled_class_dist)]
        table = wandb.Table(data=data, columns=["Class", "Distribution"])
        wandb.log({f"Labeled Class Distribution - "
                   f"{labeled_ratio:.2f}%": wandb.plot.bar(table, "Class", "Distribution")}, step=0)

        # Filter and log the class labeling progress
        class_labeling_progress = np.delete(class_labeling_progress, ignore_index)
        data = [[name, value] for name, value in zip(label_names, class_labeling_progress)]
        table = wandb.Table(data=data, columns=["Class", "Labeling Progress"])
        wandb.log(
            {f"Class Labeling Progress - "
             f"{labeled_ratio:.2f}%": wandb.plot.bar(table, "Class", "Labeling Progress")}, step=0)

        if save:
            statistics_artifact = cfg.active.statistics
            metadata = {'labeled_ratio': labeled_ratio}
            dataset_statistics = {'class_distribution': class_dist,
                                  'labeled_class_distribution': labeled_class_dist,
                                  'class_labeling_progress': class_labeling_progress,
                                  'labeled_ratio': labeled_ratio}

            torch.save(dataset_statistics, f'data/{statistics_artifact.name}.pt')
            artifact = wandb.Artifact(statistics_artifact.name, type='statistics', metadata=metadata,
                                      description='Dataset statistics')
            artifact.add_file(f'data/{statistics_artifact.name}.pt')
            wandb.run.log_artifact(artifact)

        # ---------------------------------------------------------
        # ================== MOST LABELED SAMPLE ==================
        # ---------------------------------------------------------

        most_labeled_sample, sample_labeled_ratio, label_mask = dataset.get_most_labeled_sample()
        scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

        # Open the scan and the label
        scan.open_scan(dataset.scan_files[most_labeled_sample])
        scan.open_label(dataset.label_files[most_labeled_sample])

        # Create the point cloud and the projection with fully labeled points
        cloud = np.concatenate([scan.points, scan.color * 255], axis=1)
        cloud_label_full = np.concatenate([scan.points, scan.sem_label_color * 255], axis=1)
        projection_label_full = scan.proj_sem_color

        # Open the label with the label mask
        scan.open_scan(dataset.scan_files[most_labeled_sample])
        scan.open_label(dataset.label_files[most_labeled_sample], label_mask)

        # Create the point cloud and the projection with the most labeled points
        cloud_label = np.concatenate([scan.points, scan.sem_label_color * 255], axis=1)
        projection_label = scan.proj_sem_color

        wandb.log({'Point Cloud': wandb.Object3D(cloud),
                   'Point Cloud Label - Full': wandb.Object3D(cloud_label_full),
                   f'Point Cloud Label ({sample_labeled_ratio:.2f})': wandb.Object3D(cloud_label),
                   'Projection': wandb.Image(scan.proj_color),
                   'Projection Label - Full': wandb.Image(projection_label_full),
                   f'Projection Label - ({sample_labeled_ratio:.2f})': wandb.Image(projection_label)}, step=0)

        return labeled_class_distribution, labeled_ratio


class RandomSuperpointSelector(BaseSuperpointSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device):
        super().__init__(dataset_path, cloud_paths, device)

    def select(self, dataset: Dataset, percentage: float):
        selection_size = self.get_selection_size(dataset, percentage)

        cloud_map = torch.tensor([], dtype=torch.long)
        superpoints = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)

        for cloud in self.clouds:
            cloud_superpoints, sizes = torch.unique(cloud.superpoint_map, return_counts=True)
            cloud_ids = torch.full((cloud_superpoints.shape[0],), cloud.id, dtype=torch.long)
            superpoints = torch.cat((superpoints, cloud_superpoints))
            superpoint_sizes = torch.cat((superpoint_sizes, sizes))
            cloud_map = torch.cat((cloud_map, cloud_ids))

        order = torch.randperm(superpoints.shape[0], device=self.device)
        superpoints, superpoint_sizes, cloud_map = superpoints[order], superpoint_sizes[order], cloud_map[order]

        superpoint_sizes = torch.cumsum(superpoint_sizes, dim=0)

        selected_superpoints = superpoints[superpoint_sizes < selection_size]
        selected_cloud_map = cloud_map[superpoint_sizes < selection_size]

        return self.get_voxel_selection(selected_superpoints, selected_cloud_map)


class AverageEntropySuperpointSelector(BaseSuperpointSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device):
        super().__init__(dataset_path, cloud_paths, device)

    def select(self, dataset: Dataset, model: nn.Module, percentage: float):
        selection_size = self.get_selection_size(dataset, percentage)

        cloud_map = torch.tensor([], dtype=torch.long)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)
        average_entropies = torch.tensor([], dtype=torch.float32)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for cloud in self.clouds:
                indices = np.where(dataset.cloud_map == cloud.path)[0]
                for i in tqdm(indices, desc='Mapping model output values to voxels'):
                    proj_image, _, proj_voxel_map, proj_path = dataset.get_item(i)
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

                    cloud.add_predictions(model_output.cpu(), sample_voxel_map, mc_dropout=False)

                entropies, cloud_superpoint_map, cloud_superpoint_sizes, cloud_ids = cloud.get_average_entropies()
                average_entropies = torch.cat((average_entropies, entropies))
                superpoint_map = torch.cat((superpoint_map, cloud_superpoint_map))
                superpoint_sizes = torch.cat((superpoint_sizes, cloud_superpoint_sizes))
                cloud_map = torch.cat((cloud_map, cloud_ids))

                cloud.reset()

                order = torch.argsort(average_entropies, descending=True)
                superpoint_map = superpoint_map[order]
                superpoint_sizes = superpoint_sizes[order]
                cloud_map = cloud_map[order]

                superpoint_sizes = torch.cumsum(superpoint_sizes, dim=0)
                selected_superpoints = superpoint_map[superpoint_sizes < selection_size].cpu()
                selected_cloud_map = cloud_map[superpoint_sizes < selection_size].cpu()

                return self.get_voxel_selection(selected_superpoints, selected_cloud_map)


class ViewpointVarianceSuperpointSelector(BaseSuperpointSelector):
    def __init__(self, dataset_path: str, cloud_paths: np.ndarray, device: torch.device):
        super().__init__(dataset_path, cloud_paths, device)

    def select(self, dataset: Dataset, model: nn.Module, percentage: float):

        # ----------------------------------------------------------------------
        # =========================== Initialization ===========================
        # ----------------------------------------------------------------------

        selection_size = self.get_selection_size(dataset, percentage)

        cloud_map = torch.tensor([], dtype=torch.long)
        superpoint_map = torch.tensor([], dtype=torch.long)
        superpoint_sizes = torch.tensor([], dtype=torch.long)
        average_entropies = torch.tensor([], dtype=torch.float32)

        # ----------------------------------------------------------------------------
        # =========================== Model output mapping ===========================
        # ----------------------------------------------------------------------------

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for cloud in self.clouds:
                indices = np.where(dataset.cloud_map == cloud.path)[0]
                for i in tqdm(indices, desc='Mapping model output values to voxels'):
                    proj_image, _, proj_voxel_map, proj_path = dataset.get_item(i)
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

                    cloud.add_predictions(model_output.cpu(), sample_voxel_map, mc_dropout=False)

                entropies, cloud_superpoint_map, cloud_superpoint_sizes, cloud_ids = cloud.get_average_entropies()
                average_entropies = torch.cat((average_entropies, entropies))
                superpoint_map = torch.cat((superpoint_map, cloud_superpoint_map))
                superpoint_sizes = torch.cat((superpoint_sizes, cloud_superpoint_sizes))
                cloud_map = torch.cat((cloud_map, cloud_ids))

                cloud.reset()

                order = torch.argsort(average_entropies, descending=True)
                superpoint_map = superpoint_map[order]
                superpoint_sizes = superpoint_sizes[order]
                cloud_map = cloud_map[order]

                superpoint_sizes = torch.cumsum(superpoint_sizes, dim=0)
                selected_superpoints = superpoint_map[superpoint_sizes < selection_size].cpu()
                selected_cloud_map = cloud_map[superpoint_sizes < selection_size].cpu()

                return self.get_voxel_selection(selected_superpoints, selected_cloud_map)
