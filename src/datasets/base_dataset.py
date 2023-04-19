import os
import logging

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from src.utils import load_dataset, ScanInterface, CloudInterface

log = logging.getLogger(__name__)


class Dataset(TorchDataset):
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

        super().__init__()
        assert split in ['train', 'val']

        self.cfg = cfg
        self.resume = resume
        self.split = split
        self.path = dataset_path
        self.num_clouds = num_clouds
        self.project_name = project_name
        self.al_experiment = al_experiment
        self.selection_mode = selection_mode

        self.label_map = cfg.learning_map
        self.num_classes = cfg.num_classes
        self.ignore_index = cfg.ignore_index

        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = cfg.split[split]

        self.proj_W = cfg.projection.W
        self.proj_H = cfg.projection.H
        self.proj_fov_up = cfg.projection.fov_up
        self.proj_fov_down = cfg.projection.fov_down

        self.cloud_map = None
        self.scan_files = None
        self.scan_id_map = None
        self.scan_sequence_map = None
        self.scan_selection_mask = None

        self.cloud_files = None
        self.cloud_id_map = None
        self.cloud_sequence_map = None
        self.cloud_selection_mask = None

        self.SI = ScanInterface(self.project_name, self.label_map)
        self.CI = CloudInterface(self.project_name, self.label_map)

        self.__initialize()
        self.__reduce_dataset()
        log.info(self.__repr__())

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    @property
    def clouds(self):
        selected = np.where(self.cloud_selection_mask == 1)[0]
        return self.cloud_files[selected]

    @property
    def scans(self):
        if self.selection_mode:
            return self.scan_files
        selected = np.where(self.scan_selection_mask == 1)[0]
        return self.scan_files[selected]

    @property
    def num_scans(self):
        size = 0
        for cloud_file in self.cloud_files[:self.num_clouds]:
            file_name = os.path.basename(cloud_file)
            bounds = file_name.split('.')[0].split('_')
            size += int(bounds[1]) - int(bounds[0]) + 1
        return size

    @property
    def statistics(self) -> dict:
        stats = {'class_distribution': None,
                 'labeled_ratio': None,
                 'class_progress': None,
                 'labeled_class_distribution': None,
                 'labeled_voxels': None}

        counter, labeled_counter = 0, 0
        class_counts = np.zeros(self.num_classes, dtype=np.long)
        labeled_class_counts = np.zeros(self.num_classes, dtype=np.long)

        for path in tqdm(self.clouds, desc='Calculating dataset statistics'):
            labels = self.CI.read_labels(path)
            label_mask = self.CI.read_selected_labels(path)
            sel_labels = labels[label_mask]

            class_counts, counter = self.__add_counts(labels=labels,
                                                      counter=counter,
                                                      class_counts=class_counts)
            labeled_class_counts, labeled_counter = self.__add_counts(labels=sel_labels,
                                                                      counter=labeled_counter,
                                                                      class_counts=labeled_class_counts)

        stats['labeled_voxels'] = labeled_counter
        stats['labeled_ratio'] = labeled_counter / (counter + 1e-6)
        stats['class_distribution'] = class_counts / (counter + 1e-6)
        stats['class_progress'] = labeled_class_counts / (class_counts + 1e-6)
        stats['labeled_class_distribution'] = labeled_class_counts / (labeled_counter + 1e-6)
        return stats

    @property
    def most_labeled_sample(self) -> tuple[int, float, np.ndarray]:
        idx, max_label_ratio = 0, 0
        sample_label_mask = None

        for i, scan_file in enumerate(tqdm(self.scan_files, desc='Finding most labeled sample')):
            label_mask = self.SI.read_selected_labels(scan_file)
            label_ratio = np.sum(label_mask) / len(label_mask)

            if label_ratio > max_label_ratio:
                idx, max_label_ratio = i, label_ratio
                sample_label_mask = label_mask

        return idx, max_label_ratio, sample_label_mask

    def cloud_id_of_scan(self, scan_idx: int) -> int:
        cloud = self.cloud_map[scan_idx]
        return np.where(self.cloud_files == cloud)[0][0]

    def is_scan_end_of_cloud(self, scan_idx: int) -> bool:
        if scan_idx == len(self.scan_files) - 1:
            return True
        cloud = self.cloud_map[scan_idx]
        return self.cloud_map[scan_idx + 1] != cloud

    def cloud_index(self, cloud_path: str) -> int:
        return np.where(self.cloud_files == cloud_path)[0][0]

    def label_voxels(self, voxels: np.ndarray, cloud_path: str) -> None:
        scans = self.scan_files[np.where(self.cloud_map == cloud_path)[0]]
        indices = self.scan_id_map[np.where(self.cloud_map == cloud_path)[0]]

        for scan_file, sample_idx in tqdm(zip(scans, indices), total=len(scans), desc='Labeling voxels'):
            self.scan_selection_mask[sample_idx] = self.SI.select_voxels(scan_file, voxels)

        cloud_idx = self.cloud_index(cloud_path)
        self.CI.select_voxels(cloud_path, voxels)
        self.cloud_selection_mask[cloud_idx] = True

    def __initialize(self):
        load_args = (self.path, self.project_name, self.sequences, self.split, self.al_experiment, self.resume)
        loaded_data = load_dataset(*load_args)
        self.scan_files = loaded_data['scans']
        self.cloud_map, self.scan_sequence_map = loaded_data['cloud_map'], loaded_data['scan_sequence_map']
        self.cloud_files, self.cloud_sequence_map = loaded_data['clouds'], loaded_data['cloud_sequence_map']

        self.scan_id_map = np.arange(len(self.scan_files), dtype=np.int32)
        self.cloud_id_map = np.arange(len(self.cloud_files), dtype=np.int32)

        if self.al_experiment:
            self.scan_selection_mask = np.zeros_like(self.scan_files, dtype=bool)
            self.cloud_selection_mask = np.zeros_like(self.cloud_files, dtype=bool)
        else:
            self.scan_selection_mask = np.ones_like(self.scan_files, dtype=bool)
            self.cloud_selection_mask = np.ones_like(self.cloud_files, dtype=bool)

    def __reduce_dataset(self) -> None:
        self.cloud_map = self.cloud_map[:self.num_scans]
        self.scan_files = self.scan_files[:self.num_scans]
        self.scan_id_map = self.scan_id_map[:self.num_scans]
        self.scan_sequence_map = self.scan_sequence_map[:self.num_scans]
        self.scan_selection_mask = self.scan_selection_mask[:self.num_scans]

        self.cloud_files = self.cloud_files[:self.num_clouds]
        self.cloud_id_map = self.cloud_id_map[:self.num_clouds]
        self.cloud_sequence_map = self.cloud_sequence_map[:self.num_clouds]
        self.cloud_selection_mask = self.cloud_selection_mask[:self.num_clouds]

    def __reduce_arrays(self, arrays: tuple) -> list:
        return [array[:self.num_scans] for array in arrays]

    def __add_counts(self, labels: np.ndarray, class_counts: np.ndarray,
                     counter: int) -> tuple[np.ndarray, float]:
        labels = labels[labels != self.ignore_index]
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        class_counts[unique_labels] += label_counts
        counter += np.sum(label_counts)
        return class_counts, counter
