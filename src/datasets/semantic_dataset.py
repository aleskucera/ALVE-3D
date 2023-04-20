import numpy as np
from omegaconf import DictConfig

from .base_dataset import Dataset
from src.utils import project_points, augment_points


class SemanticDataset(Dataset):
    def __init__(self,
                 split: str,
                 cfg: DictConfig,
                 dataset_path: str,
                 project_name: str,
                 resume: bool = False,
                 num_clouds: int = None,
                 sequences: iter = None,
                 al_experiment: bool = False,
                 selection_mode: bool = False):

        super().__init__(split, cfg, dataset_path,
                         project_name, resume, num_clouds,
                         sequences, al_experiment, selection_mode)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
        scan_data = self.SI.read_scan(self.scans[idx])
        points, colors, remissions = scan_data['points'], scan_data['colors'], scan_data['remissions']
        labels, voxel_map, label_mask = scan_data['labels'], scan_data['voxel_map'], scan_data['selected_labels']

        if self.selection_mode:
            voxel_map[labels == self.ignore_index] = -1

        # Augment data and apply label mask
        elif self.split == 'train':
            labels *= label_mask
            points, drop_mask = augment_points(points,
                                               drop_prob=0.5,
                                               flip_prob=0.5,
                                               rotation_prob=0.5,
                                               translation_prob=0.5)
            labels = labels[drop_mask]
            voxel_map = voxel_map[drop_mask]
            remissions = remissions[drop_mask]
            colors = colors[drop_mask] if colors is not None else None

        # Project points to image and map the projection
        proj = project_points(points, self.proj_H, self.proj_W, self.proj_fov_up, self.proj_fov_down)
        proj_distances, proj_idx, proj_mask = proj['depth'], proj['idx'], proj['mask']

        # Project remissions
        proj_remissions = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remissions[proj_mask] = remissions[proj_idx[proj_mask]]

        # Project labels
        proj_labels = np.zeros((self.proj_H, self.proj_W), dtype=np.long)
        proj_labels[proj_mask] = labels[proj_idx[proj_mask]]

        # Project voxel map
        proj_voxel_map = np.full((self.proj_H, self.proj_W), -1, dtype=np.int64)
        proj_voxel_map[proj_mask] = voxel_map[proj_idx[proj_mask]]

        if colors is None:
            proj_scan = np.concatenate([proj_distances[..., np.newaxis],
                                        proj_remissions[..., np.newaxis]],
                                       axis=-1, dtype=np.float32).transpose((2, 0, 1))
        else:
            proj_colors = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
            proj_colors[proj_mask] = colors[proj_idx[proj_mask]]

            proj_scan = np.concatenate([proj_distances[..., np.newaxis],
                                        proj_remissions[..., np.newaxis],
                                        proj_colors], axis=-1, dtype=np.float32).transpose((2, 0, 1))

        cloud_id = self.cloud_id_of_scan(idx)
        end_of_cloud = self.is_scan_end_of_cloud(idx)

        return proj_scan, proj_labels, proj_voxel_map, cloud_id, end_of_cloud

    def __len__(self):
        return len(self.scans)

    def __str__(self):
        ret = f'\n\n{self.__class__.__name__} ({self.split}):' \
              f'\n\t- Dataset size: {self.__len__()} / {self.scan_files}' \
              f'\n\t- Project name: {self.project_name}'
        if self.al_experiment:
            action = 'Selection' if self.selection_mode else 'Training'
            ret += f'\n\t- Usage: Active Learning - {action}'
        else:
            ret += f'\n\t- Usage: Full Training\n'
        return ret
