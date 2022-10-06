import numpy as np
from omegaconf import DictConfig

from .laserscan import SemLaserScan


def map_labels(label: np.ndarray, label_map: dict, inverse=False) -> np.ndarray:
    """ Map the labels to new ones based on label map
    :param label: label
    :param label_map: dictionary containing the mapping
    :param inverse: if True, the mapping will be inverted
    :return: function that maps the labels
    """
    mapped_label = np.copy(label)
    if inverse:
        for k, v in label_map.items():
            mapped_label[label == v] = k
    else:
        for k, v in label_map.items():
            mapped_label[label == k] = v
    return mapped_label


def create_semantic_laser_scan(cfg: DictConfig) -> SemLaserScan:
    """ Create a semantic laser scan object
    :param cfg: configuration
    :return: semantic laser scan object
    """
    scan = SemLaserScan(
        nclasses=len(cfg.labels),
        sem_color_dict=cfg.color_map,
        project=True,
        H=cfg.laser_scan.H,
        W=cfg.laser_scan.W,
        fov_up=cfg.laser_scan.fov_up,
        fov_down=cfg.laser_scan.fov_down)
    return scan


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """ Transform the points with the pose
    :param points: points to transform
    :param pose: pose
    :return: transformed points
    """
    hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.matmul(hom_points, pose.T)
    return transformed_points[:, :3]


def map_colors(label: np.ndarray, color_map: dict) -> np.ndarray:
    """ Map the labels to colors based on color map
    :param label: label
    :param color_map: dictionary containing the mapping
    :return: function that maps the labels
    """
    mapped_colors = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k, v in color_map.items():
        if k != 10:
            mapped_colors[label == k] = v[::-1]
    return mapped_colors
