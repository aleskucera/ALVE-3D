import logging

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mpl_toolkits.axes_grid1 import ImageGrid

from src.datasets import SemanticDataset
from src.utils.cloud import visualize_cloud, augment_points, transform_points
from src.laserscan import LaserScan, ScanVis
from src.utils.io import CloudInterface, ScanInterface
from src.utils.map import map_colors
from src.utils.visualize import bar_chart
from src.utils.project import project_points
from src.utils.map import colorize_values
from src.utils.filter import filter_scan

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


def visualize_scan_mapping(cfg: DictConfig):
    split = cfg.split if 'split' in cfg else 'train'
    log.info(f'Visualizing mapping of {split} split')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)

    SI = ScanInterface(label_map=cfg.ds.learning_map)
    CI = CloudInterface(label_map=cfg.ds.learning_map)

    if cfg.ds.name == 'SemanticKITTI':
        scan_file = dataset.scans[44]
        cloud_file = dataset.cloud_map[44]
    elif cfg.ds.name == 'KITTI360':
        scan_file = dataset.scans[491]
        cloud_file = dataset.cloud_map[491]
    else:
        raise ValueError(f'Invalid dataset name: {cfg.ds.name}')

    scan_pose = SI.read_pose(scan_file)
    scan_points = SI.read_points(scan_file)
    scan_colors = colorize_values(scan_points[:, 2], color_map='inferno',
                                  data_range=(np.min(scan_points[:, 2]), np.max(scan_points[:, 2])))
    dist = filter_scan(scan_points, 'Distance')
    rad = filter_scan(scan_points, 'Radius')

    valid_points = np.setdiff1d(np.arange(scan_points.shape[0]), rad)
    valid_points = np.setdiff1d(valid_points, dist)

    scan_points = scan_points[valid_points]
    scan_colors = scan_colors[valid_points]

    scan_points[:, 2] += 0.5
    scan_points = transform_points(scan_points, scan_pose)

    cloud_points = CI.read_points(cloud_file)
    cloud_colors = np.full((cloud_points.shape[0], 3), fill_value=0.7)

    # Concatenate points and colors
    points = np.concatenate((scan_points, cloud_points), axis=0)
    colors = np.concatenate((scan_colors, cloud_colors), axis=0)

    visualize_cloud(points, colors)


def visualize_statistics(cfg: DictConfig):
    split = cfg.split if 'split' in cfg else 'train'
    log.info(f'Visualizing statistics of {split} split')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)

    label_names = [v for v in cfg.ds.labels_train.values() if v != 'void']
    dataset_distribution = dataset.statistics['class_distribution'][1:] * 100

    bar_chart(values=dataset_distribution, labels=label_names, value_label='Proportion [%]')


def visualize_augmentation(cfg: DictConfig):
    augmentation = cfg.augmentation if 'augmentation' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    log.info(f'Visualizing statistics of {split} split')

    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=None, sequences=None)

    scan_interface = ScanInterface(label_map=cfg.ds.learning_map)

    if cfg.ds.name == 'SemanticKITTI':
        scan = dataset.scans[44]
    elif cfg.ds.name == 'KITTI360':
        scan = dataset.scans[491]
    else:
        raise ValueError(f'Invalid dataset name: {cfg.ds.name}')

    points = scan_interface.read_points(scan)
    labels = scan_interface.read_labels(scan)

    augmented_projection = project_augmentation(cfg, points, labels, augmentation=augmentation)
    augmented_projection = map_colors(augmented_projection, cfg.ds.color_map_train)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(augmented_projection.shape[1] / 100, augmented_projection.shape[0] / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(augmented_projection, aspect='auto')
    plt.show()

    # # Project original scan
    # proj_original = project_augmentation(cfg, points, labels, augmentation=None)
    # proj_original = map_colors(proj_original, cfg.ds.color_map_train)
    #
    # # Drop points
    # proj_drop = project_augmentation(cfg, points, labels, augmentation='drop')
    # proj_drop = map_colors(proj_drop, cfg.ds.color_map_train)
    #
    # # Rotate points around z-axis
    # proj_rotate = project_augmentation(cfg, points, labels, augmentation='rotate')
    # proj_rotate = map_colors(proj_rotate, cfg.ds.color_map_train)
    #
    # # Jitter points
    # proj_jitter = project_augmentation(cfg, points, labels, augmentation='jitter')
    # proj_jitter = map_colors(proj_jitter, cfg.ds.color_map_train)
    #
    # # Flip points around x-axis
    # proj_flip = project_augmentation(cfg, points, labels, augmentation='flip')
    # proj_flip = map_colors(proj_flip, cfg.ds.color_map_train)

    # fig = plt.figure(figsize=(6, 4), dpi=350)
    # grid = ImageGrid(fig, 111, nrows_ncols=(5, 1), axes_pad=0.4)
    #
    # images = [proj_original, proj_drop, proj_rotate, proj_jitter, proj_flip]
    # titles = ['Original', 'Drop Random Points', 'Rotate around z-axis', 'Jitter', 'Flip around x-axis']
    #
    # for ax, image, title in zip(grid, images, titles):
    #     ax.set_title(title)
    #     ax.imshow(image, aspect='auto')
    #     ax.axis('off')
    #
    # plt.show()


def project_augmentation(cfg, points: np.ndarray, labels: np.ndarray, augmentation: str = None):
    assert augmentation in [None, 'drop', 'rotate', 'jitter', 'flip']
    proj_H = cfg.ds.projection.H
    proj_W = cfg.ds.projection.W
    proj_fov_up = cfg.ds.projection.fov_up
    proj_fov_down = cfg.ds.projection.fov_down

    if augmentation == 'drop':
        points, drop_mask = augment_points(points, drop_prob=0.5, flip_prob=0, rotation_prob=0, translation_prob=0)
        labels = labels[drop_mask]
    elif augmentation == 'rotate':
        points, _ = augment_points(points, drop_prob=0, flip_prob=0, rotation_prob=1, translation_prob=0)
    elif augmentation == 'jitter':
        points, _ = augment_points(points, drop_prob=0, flip_prob=0, rotation_prob=0, translation_prob=1)
    elif augmentation == 'flip':
        points, _ = augment_points(points, drop_prob=0, flip_prob=1, rotation_prob=0, translation_prob=0)

    proj = project_points(points, proj_H, proj_W, proj_fov_up, proj_fov_down)
    proj_idx, proj_mask = proj['idx'], proj['mask']

    proj_labels = np.zeros((proj_H, proj_W), dtype=np.long)
    proj_labels[proj_mask] = labels[proj_idx[proj_mask]]

    return proj_labels
