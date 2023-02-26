#!/usr/bin/env python3
import os
import random

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from .project import project_scan
from .colormap import map_color, dict_to_color_map, instances_color_map


def arg_check(func):
    """ Decorator to check if the arguments are valid.
    Used in LaserScan class. The validation is done for the following types:
    - torch.Tensor: Converted to numpy array.
    - str (indicates file): Check if the file exists.
    """

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
    """ LaserScan class to store and manipulate 3D point clouds.
    The class can be used to load point clouds from SemanticKITTI format.
    It also provides methods to project the point cloud into a 2D image, and
    it can be used in ScanVis to visualize the point cloud.

    :param label_map: Dictionary with label map for training.
    :param color_map: Dictionary with color map for semantic labels mapped to training labels.
    :param colorize: If True, the point cloud will be colored according to the corresponding color maps.
    :param H: Height of the projected image.
    :param W: Width of the projected image.
    :param fov_up: Field of view up in degrees.
    :param fov_down: Field of view down in degrees.
    """

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
        self.drop_mask = None

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

        # Superpoints
        self.superpoints = None
        self.superpoints_color = None

        # Projected superpoints
        self.proj_superpoints = None
        self.proj_superpoints_color = None

    def __len__(self):
        return self.points.shape[0]

    # =======================================================================
    # ------------------------------  RAW DATA  ------------------------------
    # =======================================================================

    @arg_check
    def open_scan(self, filename: str, flip_prob: float = 0,
                  trans_prob: float = 0, rot_prob: float = 0, drop_prob: float = 0) -> None:
        """ Open scan from file. If the file is a .bin file, it is assumed to be in the SemanticKITTI
        format (x, y, z, remission).
        If the file is a .npy file, it is assumed to be a numpy array with (x, y, z, remission, r, g, b).

        :param filename: Path to the file.
        :param flip_prob: Probability of flipping the point cloud along the x-axis.
        :param trans_prob: Probability of translating the point cloud.
        :param rot_prob: Probability of rotating the point cloud around the z-axis.
        :param drop_prob: Probability of points to be dropped.
        """

        # Read scan from file
        if filename.endswith('.bin'):
            scan = np.fromfile(filename, dtype=np.float32)
            scan = scan.reshape((-1, 4))
        elif filename.endswith('.npy'):
            scan = np.load(filename)
        else:
            raise ValueError('Invalid file extension')

        # Parse scan
        points = scan[:, :3]
        remissions = scan[:, 3]
        if scan.shape[1] == 7:
            color = scan[:, 4:7]
        else:
            color = None

        self.set_scan(points, remissions, color, flip_prob, trans_prob, rot_prob, drop_prob)

    @arg_check
    def set_scan(self, points: np.ndarray, remissions: np.ndarray = None, color: np.ndarray = None,
                 flip_prob: float = 0, trans_prob: float = 0, rot_prob: float = 0, drop_prob: float = 0) -> None:
        """ Set scan from numpy arrays.

        :param points: Numpy array with shape (N, 3) containing the point cloud.
        :param remissions: Numpy array with shape (N,) containing the remission values.
        :param color: Numpy array with shape (N, 3) containing the color values.
        :param flip_prob: Probability of flipping the point cloud along the x-axis.
        :param trans_prob: Probability of translating the point cloud.
        :param rot_prob: Probability of rotating the point cloud around the z-axis.
        :param drop_prob: Probability of points to be dropped.
        """

        self.points = points
        self.remissions = remissions

        if remissions is None:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        self.radius = np.linalg.norm(self.points, axis=1)

        # Flip points
        if random.random() < flip_prob:
            self.points[:, 0] *= -1

        # Translate points
        if random.random() < trans_prob:
            self.points[:, 0] += random.uniform(-5, 5)
            self.points[:, 1] += random.uniform(-3, 3)
            self.points[:, 2] += random.uniform(-1, 0)

        # Rotate points
        if random.random() < rot_prob:
            deg = random.uniform(-180, 180)
            rot = R.from_euler('z', deg, degrees=True)
            self.points = rot.apply(self.points)

        # Drop points
        rng = np.random.default_rng()
        self.drop_mask = rng.choice([False, True], size=self.points.shape[0], p=[drop_prob, 1 - drop_prob])
        self.points = self.points[self.drop_mask]
        self.radius = self.radius[self.drop_mask]
        self.remissions = self.remissions[self.drop_mask]

        # Project data
        projection = project_scan(self.points, self.proj_H,
                                  self.proj_W, self.proj_fov_up, self.proj_fov_down)

        self.proj_idx = projection['idx']
        self.proj_xyz = projection['xyz']
        self.proj_mask = projection['mask']
        self.proj_depth = projection['depth']
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        self.proj_remission[self.proj_mask] = self.remissions[self.proj_idx[self.proj_mask]]

        if color is None and self.colorize:
            rem_range = (np.min(self.remissions), np.max(self.remissions))
            self.color = map_color(self.remissions, color_map='twilight', data_range=rem_range)
            self.proj_color = self.color[self.proj_idx]
        elif color is not None:
            self.color = color[self.drop_mask]
            self.proj_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
            self.proj_color[self.proj_mask] = self.color[self.proj_idx[self.proj_mask]]
        else:
            self.color = None
            self.proj_color = None

        self.sem_label = None
        self.sem_label_color = None
        self.inst_label = None
        self.inst_label_color = None

        self.proj_sem_label = None
        self.proj_sem_color = None
        self.proj_inst_label = None
        self.proj_inst_color = None

    # ===============================================================================
    # ------------------------------  SEMANTIC LABELS  ------------------------------
    # ===============================================================================

    @arg_check
    def open_label(self, filename: str) -> None:
        """ Open labels from a file. The file can be either a .label file or a .npy file.
        Expected format is int32, where the lower 16 bits are the semantic label and the upper 16 bits
        are the instance id.

        It is necessary to load the corresponding scan first before loading the labels to determine
        the drop mask.

        :param filename: Path to the label file.
        """

        # Read label from file
        if filename.endswith('.label'):
            label = np.fromfile(filename, dtype=np.int32)
        elif filename.endswith('.npy'):
            label = np.load(filename)
        else:
            raise ValueError('Invalid file extension')

        # Parse label
        semantics = label & 0xFFFF  # semantic label in lower half
        instances = label >> 16  # instance id in upper half

        # Map labels to train ids
        label_map = np.zeros(max(self.label_map.keys()) + 1, dtype=np.uint8)
        for label, value in self.label_map.items():
            label_map[label] = value
        semantics = label_map[semantics]

        semantics = semantics.flatten()
        instances = instances.flatten()

        # Set attributes
        self.set_label(semantics, instances)

    @arg_check
    def set_label(self, semantics: np.ndarray, instances: np.ndarray = None) -> None:
        """ Set the semantic and instance labels from numpy arrays.

        It is necessary to load the corresponding scan first
        to determine the drop mask.

        :param semantics: Numpy array with shape (N,) containing the semantic labels.
        :param instances: Numpy array with shape (N,) containing the instance ids.
        """

        assert self.sem_label is None and self.inst_label is None, 'Labels already set'

        # Set attributes, so they correspond to the points in the scan
        self.sem_label = semantics[self.drop_mask]
        self.inst_label = instances[self.drop_mask]

        if instances is None:
            self.inst_label = np.zeros((self.points.shape[0]), dtype=np.int32)

        # Project labels
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

    # ===========================================================================
    # ------------------------------  PREDICTIONS  ------------------------------
    # ===========================================================================

    @arg_check
    def open_prediction(self, filename: str) -> None:

        # Read prediction from file
        pred = np.fromfile(filename, dtype=np.int32)

        # Set attributes
        self.set_prediction(pred)

    @arg_check
    def set_prediction(self, pred: np.ndarray) -> None:
        self.proj_pred = pred

        # Set predictions for point cloud
        self.pred = np.zeros((self.points.shape[0]), dtype=np.int32)
        self.pred[self.proj_idx[self.proj_mask]] = pred[self.proj_mask]

        # Zero out predictions for points that are unlabeled
        self.pred[self.sem_label == 0] = 0
        self.proj_pred[self.proj_sem_label == 0] = 0

        # Colorize prediction
        if self.colorize:
            self.pred_color = self.color_map[self.pred]
            self.proj_pred_color = self.color_map[self.proj_pred]

    # ------------------------------  ENTROPY ------------------------------
    @arg_check
    def open_entropy(self, filename: str) -> None:

        # Read entropy from file
        entropy = np.fromfile(filename, dtype=np.float32)

        # Set attributes
        self.set_entropy(entropy)

    @arg_check
    def set_entropy(self, entropy: np.ndarray) -> None:
        self.proj_entropy = entropy

        # Set attributes, so they correspond to the points in the scan
        self.entropy = np.zeros((self.points.shape[0]), dtype=np.float32)
        self.entropy[self.proj_idx[self.proj_mask]] = entropy[self.proj_mask]

        # Zero out predictions for points that are unlabeled
        self.entropy[self.sem_label == 0] = 0
        self.proj_entropy[self.proj_sem_label == 0] = 0

        # Colorize entropy
        if self.colorize:
            entropy_range = (np.min(self.entropy), np.max(self.entropy))
            self.entropy_color = map_color(self.entropy, color_map='jet', data_range=entropy_range)
            self.proj_entropy_color = map_color(self.proj_entropy, color_map='jet', data_range=entropy_range)

    # ------------------------------  SUPERPOINTS ------------------------------

    @arg_check
    def open_superpoints(self, filename: str) -> None:

        # Read npy file
        superpoints = np.load(filename)

        # Set attributes
        self.set_superpoints(superpoints)

    @arg_check
    def set_superpoints(self, superpoints: np.ndarray) -> None:
        self.superpoints = superpoints

        # Project superpoints
        self.proj_superpoints = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_superpoints[self.proj_mask] = self.superpoints[self.proj_idx[self.proj_mask]]

        # Zero out superpoints for points that are unlabeled
        self.superpoints[self.sem_label == 0] = 0
        self.proj_superpoints[self.proj_sem_label == 0] = 0

        # Colorize superpoints
        if self.colorize:
            sp_range = (np.min(self.superpoints), np.max(self.superpoints))
            self.superpoints_color = map_color(self.superpoints, color_map='jet', data_range=sp_range)
            self.proj_superpoints_color = map_color(self.proj_superpoints, color_map='jet', data_range=sp_range)
