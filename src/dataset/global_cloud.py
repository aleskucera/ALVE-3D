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

STEP = 5
VOXEL_SIZE = 0.5


def create_global_cloud(cfg: DictConfig, sequence: int, path: str) -> None:
    cloud, color = [], []

    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=[sequence], cfg=cfg.ds, split=None)
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    print(f'Creating global cloud for sequence {sequence}...')
    print(f'Number of scans: {len(dataset.points)}')
    print(f'Number of labels: {len(dataset.labels)}')
    print(f'Number of poses: {len(dataset.poses)}')

    # Get global cloud
    points = [dataset.points[i] for i in range(len(dataset.points)) if dataset.sequence_indices[i] == sequence][::STEP]
    labels = [dataset.labels[i] for i in range(len(dataset.labels)) if dataset.sequence_indices[i] == sequence][::STEP]
    poses = [dataset.poses[i] for i in range(len(dataset.poses)) if dataset.sequence_indices[i] == sequence][::STEP]

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

    cloud = np.asarray(pcd.points)
    color = np.asarray(pcd.colors)

    np.savez(path, cloud=cloud, color=color)
