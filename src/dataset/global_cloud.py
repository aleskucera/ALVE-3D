import logging

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from src.laserscan import LaserScan
from .dataset import SemanticDataset

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: Can't import open3d.")

STEP = 10
VOXEL_SIZE = 0.5

log = logging.getLogger(__name__)


def create_global_cloud(cfg: DictConfig, sequence: int, path: str) -> None:
    """ Create a global point cloud for a given sequence.

    :param cfg: Configuration object.
    :param sequence: Sequence number.
    :param path: Path to save the global cloud.
    """
    cloud, color = [], []

    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=[sequence], cfg=cfg.ds, split=None)
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    log.info(f'Creating global cloud for sequence {sequence}...')

    # Get global cloud
    points = [dataset.scans[i] for i in range(len(dataset.scans)) if dataset.sequence_indices[i] == sequence]
    labels = [dataset.labels[i] for i in range(len(dataset.labels)) if dataset.sequence_indices[i] == sequence]
    poses = [dataset.poses[i] for i in range(len(dataset.poses)) if dataset.sequence_indices[i] == sequence]

    print('\nSequence information:')
    print(f'\tNumber of scans: {len(points)}')
    print(f'\tNumber of labels: {len(labels)}')
    print(f'\tNumber of poses: {len(poses)}')

    print(f'\nFiltering scans with step size of {STEP}...\n')

    points = points[::STEP]
    labels = labels[::STEP]
    poses = poses[::STEP]

    for p, l, pos in tqdm(zip(points, labels, poses), total=len(points)):
        scan.open_scan(p)
        scan.open_label(l)

        # Remove points that are labeled as 'unlabeled'
        scan.points = scan.points[scan.sem_label != 0]

        hom_points = np.hstack((scan.points, np.ones((scan.points.shape[0], 1))))
        transformed_points = np.matmul(hom_points, pos.T)

        cloud.append(transformed_points[:, :3])
        color.append(scan.sem_label_color[scan.sem_label != 0])

    cloud = np.concatenate(cloud, axis=0)
    color = np.concatenate(color, axis=0)

    # Down sample global cloud with voxel grid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    cloud = np.asarray(pcd.scans)
    color = np.asarray(pcd.colors)

    np.savez(path, cloud=cloud, color=color)

    log.info(f'Global cloud for sequence {sequence} saved to {path}')
