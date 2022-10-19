import logging

import hydra
import open3d as o3d
from omegaconf import DictConfig

from src import SemanticDataset, paths_to_absolute

import numpy as np

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for visualization of the SemanticKITTI dataset. Configurations are loaded from
    the conf/config.yaml file. You can change the demo mode by changing the demo variable or running the following
    command in the terminal:

        python demo.py demo=global_cloud

    """
    cfg = paths_to_absolute(cfg)
    if cfg.demo == 'global_cloud':
        show_global_cloud(cfg)
    elif cfg.demo == 'sample':
        show_sample(cfg)
    elif cfg.demo == 'sample_formats':
        show_sample_formats(cfg)
    elif cfg.demo == 'paths':
        show_paths(cfg)
    else:
        raise ValueError('Invalid demo type.')


def show_global_cloud(cfg: DictConfig):
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
