#!/usr/bin/env python
import os
import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: Can't import open3d.")

from src import SemanticDataset, set_paths, LaserScan, ScanVis, create_global_cloud, create_superpoints, \
    visualize_kitti360_conversion, convert_kitti360

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'paths':
        show_paths(cfg)
    elif cfg.action == 'dataset':
        show_dataset(cfg)
    elif cfg.action == 'global_cloud':
        show_global_cloud(cfg)
    elif cfg.action == 'superpoints':
        superpoints(cfg)
    elif cfg.action == 'kitti360_conversion_vis':
        kitti360_conversion_vis(cfg)
    elif cfg.action == 'kitti360_conversion':
        kitti360_conversion(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def show_paths(cfg: DictConfig) -> None:
    print('\nPaths dynamically generated to DictConfig object:')
    for name, path in cfg.path.items():
        print(f'\t{name}: {path}')
    print('')


def show_dataset(cfg: DictConfig) -> None:
    # dataset attributes
    size = None
    sequences = [0]
    indices = None

    # create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=sequences, cfg=cfg.ds,
                              split='train', indices=indices, size=size)

    # create semantic laser scan
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # create scan visualizer
    vis = ScanVis(scan=scan, scans=dataset.points, labels=dataset.labels, raw_cloud=True, instances=False)
    vis.run()


def show_global_cloud(cfg: DictConfig) -> None:
    sequence = 1
    file_name = f'global_cloud.npz'
    path = os.path.join(cfg.ds.path, 'sequences', f'{sequence:02d}', file_name)
    # create_global_cloud(cfg, sequence, path)

    data = np.load(path)
    cloud, color = data['cloud'], data['color']

    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])


def superpoints(cfg: DictConfig):
    sequence = 3
    number_of_superpoints = 30000

    path = os.path.join(cfg.ds.path, 'sequences', f'{sequence:02d}', 'superpoints')
    os.makedirs(path, exist_ok=True)

    # create_superpoints(cfg=cfg, sequence=sequence, num_points=number_of_superpoints, directory=path)

    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=[sequence], cfg=cfg.ds, split='train')

    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)
    vis = ScanVis(scan=scan, scans=dataset.points, superpoints=dataset.superpoints, raw_cloud=False)
    vis.run()


def kitti360_conversion_vis(cfg: DictConfig):
    seq = 0
    visualize_kitti360_conversion(cfg, seq)


def kitti360_conversion(cfg: DictConfig):
    seq = 0
    convert_kitti360(cfg, seq)


if __name__ == '__main__':
    main()
