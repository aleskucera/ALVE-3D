import logging

from omegaconf import DictConfig

from src.datasets import SemanticDataset
from src.utils.cloud import visualize_cloud
from src.laserscan import LaserScan, ScanVis
from src.utils.io import CloudInterface
from src.utils.map import map_colors
from src.utils.visualize import bar_chart

log = logging.getLogger(__name__)


def visualize_scans(cfg: DictConfig):
    split = cfg.split if 'split' in cfg else 'train'

    log.info(f'Visualizing scans of {split} split')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)

    laser_scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True,
                           H=cfg.ds.projection.H, W=cfg.ds.projection.W, fov_up=cfg.ds.projection.fov_up,
                           fov_down=cfg.ds.projection.fov_down)

    vis = ScanVis(laser_scan=laser_scan, scans=dataset.scan_files, labels=dataset.scan_files,
                  raw_scan=True)
    vis.run()


def visualize_clouds(cfg: DictConfig):
    color_arg = cfg.color if 'color' in cfg else 'labels'
    split = cfg.split if 'split' in cfg else 'train'

    log.info(f'Visualizing clouds of {split} split')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)
    cloud_interface = CloudInterface(label_map=cfg.ds.learning_map)

    for cloud_file in dataset.clouds:
        points = cloud_interface.read_points(cloud_file)
        labels = cloud_interface.read_labels(cloud_file)
        rgb = cloud_interface.read_colors(cloud_file)

        if color_arg == 'rgb':
            colors = rgb
        elif color_arg == 'labels':
            colors = map_colors(labels, cfg.ds.color_map_train)
        else:
            raise ValueError(f'Invalid color argument: {color_arg}')
        visualize_cloud(points, colors)


def visualize_statistics(cfg: DictConfig):
    split = cfg.split if 'split' in cfg else 'train'

    log.info(f'Visualizing statistics of {split} split')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)

    label_names = [v for v in cfg.ds.labels_train.values() if v != 'void']
    dataset_distribution = dataset.statistics['class_distribution'][1:] * 100

    bar_chart(values=dataset_distribution, labels=label_names, value_label='Proportion [%]')
