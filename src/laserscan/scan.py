import random

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.project import project_points
from src.utils.map import colorize_values, map_colors, map_labels


class LaserScan:
    """ LaserScan class to store and manipulate 3D point clouds.
    The class can be used to load the data from the SemanticKITTI format
    and also form the Active Learning format. It also automatically projects
    the point cloud to an image and maps the labels to the projected image.

    The class is designed to be used mainly for visualization purposes.

    :param label_map: Dictionary with label map for training.
    :param color_map: Dictionary with color map for semantic labels mapped to
                      training labels. (default: None)
    :param colorize: If True, the point cloud will be colored according
                     to the corresponding color maps. (default: False)
    :param H: Height of the projected image. (default: 64)
    :param W: Width of the projected image. (default: 1024)
    :param fov_up: Field of view up in degrees. (default: 3.0)
    :param fov_down: Field of view down in degrees. (default: -25.0)
    """

    def __init__(self, label_map: dict, color_map: dict = None, colorize: bool = False, H=64, W=1024, fov_up=3.0,
                 fov_down=-25.0):

        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

        self.label_map = label_map

        self.colorize = colorize
        if colorize:
            assert color_map is not None, 'Color map must be provided if colorize is True'
            self.sem_color_map = color_map

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
        self.label = None
        self.label_color = None

        # Projected semantic labels
        self.proj_label = None
        self.proj_label_color = None

        # Predictions
        self.pred = None
        self.pred_color = None

        # Projected predictions
        self.proj_pred = None
        self.proj_pred_color = None

    def __len__(self):
        return self.points.shape[0]

    # =========================================================================
    # ------------------------------  SCAN DATA  ------------------------------
    # =========================================================================

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
            points = scan[:, :3]
            remissions = scan[:, 3]
            color = None
        elif filename.endswith('.h5'):
            with h5py.File(filename, 'r') as f:
                points = np.array(f['points'])
                remissions = np.array(f['remissions'])
                color = np.array(f['colors']) if 'colors' in f else None
        else:
            raise ValueError('Invalid file extension')

        self.set_scan(points, remissions, color, flip_prob, trans_prob, rot_prob, drop_prob)

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

        self.points = points.astype(np.float32)
        self.radius = np.linalg.norm(self.points, axis=1)
        self.remissions = remissions if remissions is not None else np.zeros((points.shape[0]), dtype=np.float32)

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
        projection = project_points(self.points, self.proj_H,
                                    self.proj_W, self.proj_fov_up, self.proj_fov_down)

        self.proj_idx = projection['idx']
        self.proj_xyz = projection['xyz']
        self.proj_mask = projection['mask']
        self.proj_depth = projection['depth']
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        self.proj_remission[self.proj_mask] = self.remissions[self.proj_idx[self.proj_mask]]

        if color is not None:
            self.color = color[self.drop_mask]
            self.proj_color = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
            self.proj_color[self.proj_mask] = self.color[self.proj_idx[self.proj_mask]]
        elif self.colorize:
            rem_range = (np.min(self.radius), np.max(self.radius))
            self.color = colorize_values(self.radius, color_map='turbo', data_range=rem_range)
            self.proj_color = self.color[self.proj_idx]
        else:
            self.color, self.proj_color = None, None

        self.label, self.label_color = None, None
        self.proj_label, self.proj_label_color = None, None

    # ======================================================================
    # ------------------------------  LABELS  ------------------------------
    # ======================================================================

    def open_label(self, filename: str, label_mask: np.ndarray = None) -> None:
        """ Open labels from a file. The file can be either a .label file or a .npy file.
        Expected format is int32, where the lower 16 bits are the semantic label and the upper 16 bits
        are the instance id.

        It is necessary to load the corresponding scan first before loading the labels to determine
        the drop mask.

        :param filename: Path to the label file.
        :param label_mask: Numpy array with shape (N,) containing a mask for the labels.
        """

        # Read label from file
        if filename.endswith('.label'):
            label_data = np.fromfile(filename, dtype=np.int32)
            labels = label_data & 0xFFFF  # semantic label in lower half
        elif filename.endswith('.h5'):
            with h5py.File(filename, 'r') as f:
                labels = np.asarray(f['labels'])
        else:
            raise ValueError('Invalid file extension')

        # Map labels to train ids
        labels = map_labels(labels.flatten(), self.label_map)

        # Set attributes
        self.set_label(labels, label_mask)

    def set_label(self, labels: np.ndarray, label_mask: np.ndarray = None) -> None:

        assert self.label is None, 'Labels already set'

        if label_mask is not None and np.sum(label_mask) != 0:
            labels *= label_mask.astype(bool)

        # Set attributes, so they correspond to the points in the scan
        self.label = labels[self.drop_mask]

        # Project labels
        proj_indices = self.proj_idx[self.proj_mask]
        self.proj_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_label[self.proj_mask] = self.label[proj_indices]

        # Colorize label
        if self.colorize:
            self.label_color = map_colors(self.label, self.sem_color_map)
            self.proj_label_color = map_colors(self.proj_label, self.sem_color_map)

    # ======================================================================
    # ------------------------------  LABELS  ------------------------------
    # ======================================================================

    def open_prediction(self, filename: str) -> None:

        # Read prediction from file
        with h5py.File(filename, 'r') as f:
            prediction = np.asarray(f['prediction'])

        # Set attributes
        self.set_prediction(prediction)

    def set_prediction(self, prediction: np.ndarray) -> None:
        self.proj_pred = prediction

        # Set predictions for point cloud
        self.pred = np.zeros((self.points.shape[0]), dtype=np.int32)
        self.pred[self.proj_idx[self.proj_mask]] = prediction[self.proj_mask]

        # Zero out predictions for points that are unlabeled
        self.pred[self.label == 0] = 0
        self.proj_pred[self.proj_label == 0] = 0

        # Colorize prediction
        if self.colorize:
            self.pred_color = map_colors(self.pred, self.sem_color_map)
            self.proj_pred_color = map_colors(self.proj_pred, self.sem_color_map)
