#!/usr/bin/env python

import time
import math
import random
import atexit
import logging

import hydra
# import open3d as o3d
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

from src import SemanticDataset, set_paths, check_value, \
    supervise_remote, start_tensorboard, terminate_tensorboard

import numpy as np
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


@atexit.register
def exit_function():
    """ Terminate all running tensorboard processes when the program exits. """
    terminate_tensorboard()
    log.info('Terminated all running tensorboard processes.')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for visualization of the SemanticKITTI dataset. Configurations are loaded from
    the conf/config.yaml file. You can change the demo mode by changing the demo variable or running the following
    command in the terminal:

        python demo.py action=global_cloud

    """

    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)
    log.info(f'Starting demo with action: {cfg.action}')
    check_value(cfg.node, ['master', 'slave'])
    check_value(cfg.action, ['global_cloud', 'sample', 'sample_formats', 'paths', 'simulation'])

    if cfg.action == 'global_cloud':
        show_global_cloud(cfg)
    elif cfg.action == 'sample':
        show_sample(cfg)
    elif cfg.action == 'sample_formats':
        show_sample_formats(cfg)
    elif cfg.action == 'paths':
        show_paths(cfg)
    elif cfg.action == 'simulation':
        if cfg.node == 'master':
            start_tensorboard(cfg.path.output)
            time.sleep(10)
            log.info('Starting supervisor')
            supervise_remote(cfg)
            pass
        elif cfg.node == 'slave':
            computational_simulation(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def computational_simulation(cfg: DictConfig) -> None:
    writer = SummaryWriter(cfg.path.output)
    for i in range(500):
        x = i / 10
        a = random.random()
        writer.add_scalar('Sin(x)', a * math.sin(x), i)
        writer.add_scalar('Cos(x)', a * math.cos(x), i)
        log.info(f'Iteration {i} completed')
        time.sleep(0.5)


def show_global_cloud(cfg: DictConfig):
    import open3d as o3d
    dataset = SemanticDataset(path=cfg.path.kitti, split='train', cfg=cfg.kitti)

    # Load global point cloud for visualization
    points, colors = dataset.create_global_cloud(sequence_index=2, step=40)

    # Create point cloud geometry object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with black background
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


def show_sample(cfg: DictConfig):
    import open3d as o3d
    dataset = SemanticDataset(path=cfg.path.kitti, split='train', cfg=cfg.kitti)

    # Load semantic point cloud sample for visualization
    sample = dataset.get_sem_cloud(0)

    # Create point cloud geometry object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(sample.points)
    cloud.colors = o3d.utility.Vector3dVector(sample.colors)

    # Visualize the point cloud with black background
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


def show_sample_formats(cfg: DictConfig):
    dataset = SemanticDataset(path=cfg.path.kitti, split='train', cfg=cfg.kitti)

    # Load train sample
    print('Loading train sample...')
    sample_train = dataset[0]
    print(f"\n\tx size: {sample_train[0].shape} \n"
          f"\ty size: {sample_train[1].shape}")

    # Load semantic point cloud sample for visualization
    print('Loading semantic point cloud...')
    sample_cloud = dataset.get_sem_cloud(0)
    print(sample_cloud)

    # Load depth image sample for visualization
    print("Loading depth image...")
    sample_depth = dataset.get_sem_depth(0)
    print(sample_depth)


def show_paths(cfg: DictConfig) -> None:
    for name, path in cfg.path.items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
