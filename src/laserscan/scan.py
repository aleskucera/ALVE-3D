#!/usr/bin/env python3
import os
import torch
import numpy as np
from .project import project_scan
from .colormap import map_color, dict_to_color_map, instances_color_map


def arg_check(func):
    def wrapper(self, *args, **kwargs):
        new_args = []
        new_kwargs = {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.cpu().numpy()
            elif isinstance(arg, str):
                assert os.path.isfile(arg), f'File {arg} does not exist'
            new_args.append(arg)

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            elif isinstance(value, str):
                assert os.path.isfile(value), f'File {value} does not exist'
            new_kwargs[key] = value

        func(self, *new_args, **new_kwargs)

    return wrapper


class LaserScan:
    def __init__(self, label_map, color_map=None, colorize=False, H=64, W=1024, fov_up=3.0,
                 fov_down=-25.0):
        self.proj_H = H
        self.proj_W = W

        self.colorize = colorize

        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

        self.label_map = label_map

        if colorize:
            assert color_map is not None, 'Color map must be provided if colorize is True'
            self.color_map = dict_to_color_map(color_map)
            self.inst_color_map = instances_color_map()

        # Raw cloud data
        self.points = None
        self.color = None
        self.remissions = None
        self.radius = None

        # Raw projected data
        self.proj_xyz = None
        self.proj_idx = None
        self.proj_color = None
        self.proj_depth = None
        self.proj_mask = None
        self.proj_remission = None

        # Semantic labels
        self.sem_label = None
        self.sem_label_color = None

        self.inst_label = None
        self.inst_label_color = None

        # Projected semantic labels
        self.proj_sem_label = None
        self.proj_sem_color = None

        self.proj_inst_label = None
        self.proj_inst_color = None

        # Predictions
        self.pred = None
        self.diff = None

        self.pred_color = None
        self.diff_color = None

        # Projected predictions
        self.proj_pred = None
        self.proj_diff = None

        self.proj_pred_color = None
        self.proj_diff_color = None

        # Entropy
        self.entropy = None
        self.entropy_color = None

        # Projected entropy
        self.proj_entropy = None
        self.proj_entropy_color = None

    def __len__(self):
        return self.points.shape[0]

    # ------------------------------  RAW DATA ------------------------------

    @arg_check
    def open_points(self, filename: str) -> None:
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]
        remissions = scan[:, 3]
        self.set_points(points, remissions)

    @arg_check
    def set_points(self, points: np.ndarray, remissions: np.ndarray = None) -> None:
        # Points are first 3 columns
        self.points = points

        # Remissions are the 4th column
        if remissions is not None:
            self.remissions = remissions
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        self.radius = np.linalg.norm(self.points, axis=1)

        # Project data
        projection = project_scan(self.points, self.remissions, self.proj_H,
                                  self.proj_W, self.proj_fov_up, self.proj_fov_down)

        self.proj_idx = projection['idx']
        self.proj_xyz = projection['xyz']
        self.proj_mask = projection['mask']
        self.proj_depth = projection['depth']
        self.proj_remission = projection['remission']

        # Colorize data
        if self.colorize:
            self.color = map_color(self.radius, color_map='twilight',
                                   data_range=(np.min(self.radius), np.max(self.radius)))
            self.proj_color = map_color(self.proj_depth, color_map='twilight',
                                        data_range=(np.min(self.proj_depth), np.max(self.proj_depth)))

    # ------------------------------  SEMANTIC LABELS ------------------------------
    @arg_check
    def open_label(self, filename: str) -> None:
        label = np.fromfile(filename, dtype=np.int32)
        semantics = label & 0xFFFF  # semantic label in lower half
        instances = label >> 16  # instance id in upper half

        label_map = np.zeros(max(self.label_map.keys()) + 1, dtype=np.uint8)
        for label, value in self.label_map.items():
            label_map[label] = value

        semantics = label_map[semantics]

        self.set_label(semantics, instances)

    @arg_check
    def set_label(self, semantics: np.ndarray, instances: np.ndarray) -> None:
        # semantic label in lower half
        self.sem_label = semantics

        # instance label in upper half
        self.inst_label = instances

        # Project label
        proj_indices = self.proj_idx[self.proj_mask]

        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

        self.proj_sem_label[self.proj_mask] = self.sem_label[proj_indices]
        self.proj_inst_label[self.proj_mask] = self.inst_label[proj_indices]

        # Colorize label
        if self.colorize:
            self.sem_label_color = self.color_map[self.sem_label]
            self.inst_label_color = self.inst_color_map[self.inst_label]

            self.proj_sem_color = self.color_map[self.proj_sem_label]
            self.proj_inst_color = self.inst_color_map[self.proj_inst_label]

    # ------------------------------  PREDICTIONS ------------------------------
    @arg_check
    def open_prediction(self, filename: str) -> None:
        pred = np.fromfile(filename, dtype=np.int32)
        self.set_prediction(pred)

    @arg_check
    def set_prediction(self, pred: np.ndarray) -> None:
        self.proj_pred = pred
        self.proj_pred[self.proj_sem_label == 0] = 0

        self.pred = np.zeros((self.points.shape[0]), dtype=np.int32)
        self.pred[self.proj_idx[self.proj_mask]] = pred[self.proj_mask]
        self.pred[self.sem_label == 0] = 0

        # Colorize prediction
        if self.colorize:
            self.pred_color = self.color_map[self.pred]
            self.proj_pred_color = self.color_map[self.proj_pred]

    # ------------------------------  ENTROPY ------------------------------
    @arg_check
    def open_entropy(self, filename: os.PathLike) -> None:
        entropy = np.fromfile(filename, dtype=np.float32)
        self.set_entropy(entropy)

    @arg_check
    def set_entropy(self, entropy: np.ndarray) -> None:
        self.entropy = np.zeros((self.points.shape[0]), dtype=np.float32)
        self.entropy[self.proj_idx[self.proj_mask]] = entropy[self.proj_mask]
        self.proj_entropy = entropy

        # Colorize entropy
        if self.colorize:
            self.entropy_color = map_color(self.entropy, color_map='jet')
            self.proj_entropy_color = map_color(self.proj_entropy, color_map='jet')
