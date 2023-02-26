#!/usr/bin/env python
import os
import logging

import wandb
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: Can't import open3d.")

from src import SemanticDataset, set_paths, LaserScan, ScanVis, create_global_cloud, visualize_kitti360_conversion, \
    create_config

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'hydra':
        show_hydra(cfg)
    elif cfg.action == 'dataset':
        show_dataset(cfg)
    elif cfg.action == 'log_dataset':
        log_dataset(cfg)
    elif cfg.action == 'global_cloud':
        show_global_cloud(cfg)
    elif cfg.action == 'kitti360_config':
        create_kitti360_config()
    elif cfg.action == 'kitti360_conversion':
        kitti360_conversion(cfg)
    elif cfg.action == 'superpoints':
        show_superpoints(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def show_hydra(cfg: DictConfig) -> None:
    """ Show how to use Hydra for configuration management.
    :param cfg: Configuration object.
    """
    print('\nThis project uses Hydra for configuration management.')
    print(f'Configuration object type is: {type(cfg)}')
    print(f'Configuration content is separated into groups:')
    for group in cfg.keys():
        print(f'\t{group}')

    print('\nPaths dynamically generated to DictConfig object:')
    for name, path in cfg.path.items():
        print(f'\t{name}: {path}')
    print('')


def show_dataset(cfg: DictConfig) -> None:
    """ Show how to use SemanticDataset.
    :param cfg: Configuration object.
    """

    split = None
    size = None
    sequences = [0]
    indices = None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=sequences, cfg=cfg.ds,
                              split=split, indices=indices, size=size)

    # Create scan object
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # Visualizer
    vis = ScanVis(scan=scan, scans=dataset.scans, labels=dataset.labels, raw_cloud=True, instances=False)
    vis.run()


def log_dataset(cfg: DictConfig) -> None:
    """ Log dataset statistics.
    :param cfg: Configuration object.
    """

    split = 'train'
    sequence = 3
    run_name = f'Sequence {sequence} - {split}'

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=[sequence], cfg=cfg.ds,
                              split=split)

    # Create scan object
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # Log dataset statistics
    with wandb.init(project='Sequence Visualization', name=run_name):
        # Generate random number
        i = np.random.randint(0, len(dataset))
        scan.open_scan(dataset.scans[i])
        scan.open_label(dataset.labels[i])

        cloud = np.concatenate([scan.points, scan.color * 255], axis=1)
        label = np.concatenate([scan.points, scan.sem_label_color * 255], axis=1)

        # Log statistics
        wandb.log({'Point Cloud': wandb.Object3D(cloud),
                   'Point Cloud Label': wandb.Object3D(label),
                   'Projection': wandb.Image(scan.proj_color),
                   'Projection Label': wandb.Image(scan.proj_sem_color)})

        print(f'\nLogged scan: {dataset.scans[i]} \n'
              f'Logged label: {dataset.labels[i]} \n')


def show_global_cloud(cfg: DictConfig) -> None:
    """ Create global cloud and visualize it.
    :param cfg: Configuration object.
    """

    sequence = 0
    file_name = f'global_cloud.npz'
    path = os.path.join(cfg.ds.path, 'sequences', f'{sequence:02d}', file_name)
    create_global_cloud(cfg, sequence, path)

    data = np.load(path)
    cloud, color = data['cloud'], data['color']

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])


def kitti360_conversion(cfg: DictConfig):
    """ Visualize KITTI-360 conversion.
    :param cfg: Configuration object.
    """

    visualize_kitti360_conversion(cfg)


def create_kitti360_config():
    """ Create KITTI-360 config file.
    """
    create_config()


def show_superpoints(cfg):
    # Create point cloud from .ply file
    rgb_path = "/home/ales/Datasets/FAKE_KITTI/clouds/train/0000000599_0000000846_rgb.ply"
    # superpoints_path = "/home/ales/Datasets/FAKE_KITTI/clouds/train/0000000599_0000000846_partition.ply"

    # rgb_path = "/home/ales/Datasets/S3DIS_copy/clouds/Area_1/conferenceRoom_1_partition.ply"
    superpoints_path = "/home/ales/Datasets/S3DIS/clouds/Area_1/office_19_partition.ply"

    rgb_pcd = o3d.io.read_point_cloud(rgb_path)
    superpoints_pcd = o3d.io.read_point_cloud(superpoints_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = superpoints_pcd.points
    pcd.colors = superpoints_pcd.colors

    def next_scan(vis):
        pcd.points = rgb_pcd.points
        pcd.colors = rgb_pcd.colors
        vis.update_geometry(pcd)

    def prev_scan(vis):
        pcd.points = superpoints_pcd.points
        pcd.colors = superpoints_pcd.colors
        vis.update_geometry(pcd)

    key_callbacks = {
        ord('N'): next_scan,
        ord('B'): prev_scan,
    }

    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd], key_callbacks)


if __name__ == '__main__':
    main()
