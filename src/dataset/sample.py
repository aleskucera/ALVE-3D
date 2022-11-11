import torch
import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass

from .laserscan import SemLaserScan


@dataclass
class Sample:
    # Specific attributes
    id: int
    time: float

    # Paths
    label_path: str
    points_path: str

    # Transformations
    pose: np.ndarray
    calibration: dict
    absolute_position = False

    # Data for learning
    x: torch.tensor = None
    y: torch.tensor = None

    # Point cloud data
    points: np.ndarray = None
    colors: np.ndarray = None

    # Depth image data
    depth_image: np.ndarray = None
    depth_color: np.ndarray = None

    def load_learning_data(self, scan: SemLaserScan, learning_map: dict):
        scan.open_scan(self.points_path)
        scan.open_label(self.label_path)
        self.x = torch.tensor(scan.proj_range).unsqueeze(0)
        self.y = torch.tensor(_map_labels(scan.proj_sem_label, learning_map))

    def load_semantic_depth(self, scan: SemLaserScan):
        scan.open_scan(self.points_path)
        scan.open_label(self.label_path)
        scan.colorize()
        self.depth_image = scan.proj_xyz
        self.depth_color = scan.proj_sem_color

    def load_semantic_cloud(self, scan: SemLaserScan):
        scan.open_scan(self.points_path)
        scan.open_label(self.label_path)
        scan.colorize()
        self.points = scan.points
        self.colors = scan.sem_label_color

    def load_laserscan(self, scan: SemLaserScan):
        scan.open_scan(self.points_path)
        scan.open_label(self.label_path)
        scan.colorize()

    def to_absolute_position(self) -> None:
        """Transform the points from the sensor frame to the world frame
        :return: None
        """
        if not self.absolute_position:
            rotation = self.pose[:3, :3]
            translation = self.pose[:3, 3]
            self.points = np.dot(self.points, rotation.T) + translation[np.newaxis, :]
            self.absolute_position = True

    def __repr__(self):
        output = f"\n\tId: {self.id}" \
                 f"\n\tTime: {self.time}" \
                 f"\n\tLabel_path: {self.label_path}" \
                 f"\n\tPoints_path: {self.points_path}" \
                 f"\n\tAbsolute_position: {self.absolute_position}" \
                 f"\n\t--- Train data ---" \
                 f"\n\tx: {_tensor_shape(self.x)}" \
                 f"\n\ty: {_tensor_shape(self.y)}" \
                 f"\n\t--- Depth image data ---" \
                 f"\n\tDepth_image: {_numpy_shape(self.depth_image)}" \
                 f"\n\tDepth_color: {_numpy_shape(self.depth_color)}" \
                 f"\n\t--- Point cloud data ---" \
                 f"\n\tPoints: {_numpy_shape(self.points)}" \
                 f"\n\tColors: {_numpy_shape(self.colors)}"
        return output


# ----------- Auxiliary functions ------------

def _tensor_shape(tensor: torch.tensor) -> str:
    # print(tensor.size)
    if tensor is None:
        return 'Not initialized'
    return tensor.size


def _numpy_shape(array: np.ndarray):
    if array is None:
        return 'Not initialized'
    return array.shape


def _mask_to_colors(mask: np.ndarray, color_map: dict) -> np.ndarray:
    """Convert a mask to a color mask
    :param mask: label mask
    :param color_map: dict mapping labels to colors where the key is the label and the value is the color
    :return: color mask
    """
    # Add a channel dimension
    color_mask_shape = mask.shape + (3,)
    color_mask = np.zeros(color_mask_shape, dtype=np.float)

    # Fill the colors
    for label, color in color_map.items():
        color = np.array(color, dtype=np.float) / 255
        color_mask[mask == label] = color

    return color_mask


def _colors_to_mask(color_mask: np.ndarray, color_map: dict) -> np.ndarray:
    """Convert a color mask to a mask
    :param color_mask: color mask
    :param color_map: dict mapping labels to colors where the key is the label and the value is the color
    :return: label mask
    """
    # Remove the channel dimension
    mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)

    # Fill the labels
    for label, color in color_map.items():
        mask[np.all(color_mask == color, axis=2)] = label

    return mask


def _map_labels(labels: np.ndarray, label_map: dict) -> np.ndarray:
    """Map the labels
    :param labels: label mask
    :param label_map: label map
    :return: mapped labels
    """
    label_map = OmegaConf.to_container(label_map)
    labels = np.vectorize(label_map.get)(labels)
    return labels


def _flip_data(points: np.ndarray, label: np.ndarray, prob: float = 0.5) -> tuple:
    """ Flip the data
    :param points: points
    :param label: labels
    :param prob: probability of flipping
    :return: flipped data and labels
    """
    # with probability of prob
    if np.random.random() < prob:
        points = np.flip(points, axis=2)
        label = np.flip(label, axis=1)
    return points, label
