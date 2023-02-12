import yaml
from omegaconf import DictConfig

from .labels import labels
from .converter import Kitti360Converter


def visualize_kitti360_conversion(cfg: DictConfig):
    """Visualize conversion of KITTI-360 dataset to SemanticKITTI format
    :param cfg: Config object
    """
    converter = Kitti360Converter(cfg)
    converter.visualize()


def convert_kitti360(cfg: DictConfig):
    """Convert KITTI-360 dataset to SemanticKITTI format
    :param cfg: Config object
    """
    converter = Kitti360Converter(cfg)
    converter.convert()


def create_config():
    """Create config file for KITTI-360 dataset from kitti360scripts.helpers.labels.
    The changes are:
    - learning_map"""

    name = 'kitti-360'
    path = 'data/KITTI-360'

    id_to_name = {label.id: label.name for label in labels}

    learning_map = {label.id: train_id(label) for label in labels}
    learning_map_inv = {train_id(label): label.id for label in labels}
    learning_ignore = {train_id(label): label.ignoreInEval for label in labels}

    color_map = {label.id: list(label.color) for label in labels}
    color_map_train = {train_id(label): list(label.color) for label in labels}
    color_map_train[0] = [0, 0, 0]

    # Range, intensity, r, g, b
    mean = [0, 0, 0, 0, 0]
    std = [1, 1, 1, 1, 1]

    config = {'name': name, 'path': path, 'labels': id_to_name, 'color_map_train': color_map_train,
              'color_map': color_map, 'learning_map': learning_map, 'learning_map_inv': learning_map_inv,
              'learning_ignore': learning_ignore, 'mean': mean, 'std': std}

    # create config file
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)


def train_id(label):
    if label.ignoreInEval:
        return 0
    return label.trainId + 1


def extract_kitti360_features(path: str):
    """Extract features from KITTI-360 dataset. Extracted features are: mean, std and label distribution
    :param path: Path to KITTI-360 dataset
    """
    pass
