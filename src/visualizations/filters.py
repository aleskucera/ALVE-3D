import logging

import numpy as np
from omegaconf import DictConfig
import open3d as o3d

from src.utils.io import ScanInterface
from src.datasets import SemanticDataset
from src.utils.filter import filter_scan
from src.utils.map import colorize_values
from src.utils.cloud import visualize_cloud

log = logging.getLogger(__name__)


def visualize_filters(cfg: DictConfig):
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)

    scan_interface = ScanInterface()
    points = scan_interface.read_points(dataset.scans[491])

    radius = points[:, 2]
    colors = colorize_values(radius, color_map='inferno', data_range=(np.min(radius), np.max(radius)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

    dist = filter_scan(points, 'distance')
    rad = filter_scan(points, 'radius')

    dist_colors = np.full(colors.shape, [0.7, 0.7, 0.7])
    dist_colors[dist] = np.array([131, 56, 236]) / 255
    visualize_cloud(points, dist_colors)

    rad_colors = np.full(colors.shape, [0.7, 0.7, 0.7])
    rad_colors[rad] = np.array([255, 0, 110]) / 255
    visualize_cloud(points, rad_colors)

    # Filter from points indices rad and dist
    valid_points = np.setdiff1d(np.arange(points.shape[0]), rad)
    valid_points = np.setdiff1d(valid_points, dist)
    visualize_cloud(points[valid_points], colors[valid_points])
