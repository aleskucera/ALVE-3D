from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from omegaconf import DictConfig

from .labels import labels


def create_config():
    """Create config file for KITTI-360 dataset from kitti360scripts.helpers.labels.
    The changes are:
    - learning_map"""

    config = CommentedMap()
    config['name'] = 'KITTI-360'
    config['path'] = 'data/KITTI-360'

    config['num_classes'] = 20
    config['num_semantic_channels'] = 5
    config['num_partition_channels'] = {'local': 6, 'global': 7}

    config['ignore_index'] = 0

    config['laser_scan'] = {'H': 64, 'W': 2048, 'fov_up': 16.6, 'fov_down': -16.6}

    config['sequences'] = flow_list([0, 2, 3, 4, 5, 6, 7, 9, 10])

    config['split'] = {'train': flow_list([0, 2, 3, 4, 5, 6, 7, 9, 10]),
                       'val': flow_list([0, 2, 3, 4, 5, 6, 7, 9, 10])}

    config['labels'] = {label.id: label.name for label in labels}

    config['color_map'] = {label.id: flow_list(label.color) for label in labels}
    config['color_map_train'] = {train_id(label): flow_list(label.color) for label in labels}
    config['color_map_train'][0] = flow_list([0, 0, 0])

    config['learning_map'] = {label.id: train_id(label) for label in labels}
    config['learning_map_inv'] = {train_id(label): label.id for label in labels}
    config['learning_ignore'] = {train_id(label): label.ignoreInEval for label in labels}

    config['mean'] = flow_list([0.0, 0.0, 0.0])
    config['std'] = flow_list([1.0, 1.0, 1.0])

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.dump(config, open('kitti360.yaml', 'w'))


def flow_list(items):
    seq = CommentedSeq(items)
    seq.fa.set_flow_style()
    return seq


def train_id(label):
    if label.ignoreInEval:
        return 0
    return label.trainId + 1


def save_global_cloud():
    pass
