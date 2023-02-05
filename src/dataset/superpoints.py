import os
import logging

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.cluster import MiniBatchKMeans
from omegaconf import DictConfig

from .dataset import SemanticDataset
from src.laserscan import LaserScan

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: Can't import open3d.")

log = logging.getLogger(__name__)


def create_superpoints(cfg: DictConfig, sequence: int, num_points: int, directory: str) -> None:
    assert os.path.exists(directory), f'Path {directory} does not exist.'

    # Create dataset and LaserScan object
    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=[sequence], cfg=cfg.ds, split='train')
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=False)

    # Get global cloud
    points = [dataset.points[i] for i in range(len(dataset.points)) if dataset.sequence_indices[i] == sequence]
    poses = [dataset.poses[i] for i in range(len(dataset.poses)) if dataset.sequence_indices[i] == sequence]

    # Transform points to the first frame
    log.info('Transforming points to the first frame...')
    transformed_points = transform_points(scan, points, poses)

    # Concatenate points to global point cloud and keep track of the original indices
    cloud = np.concatenate(transformed_points, axis=0)
    split_indices = np.cumsum([len(p) for p in transformed_points])[:-1]

    # Create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    # Down sample point cloud
    log.info('Down sampling point cloud...')
    down_pcd = down_sample_points(pcd)

    # Calculate features
    log.info('Calculating features...')
    features = get_features(down_pcd)

    # Cluster points
    log.info('Clustering points...')
    model = MiniBatchKMeans(n_clusters=num_points, random_state=0, batch_size=3000, verbose=True)
    cluster_indices = model.fit_predict(features)

    # Extend clusters to original point cloud
    log.info('Extending clusters to original point cloud...')
    global_superpoint_ids = extend_clusters(cloud, np.array(down_pcd.points), cluster_indices)

    # Split superpoint ids into individual scans
    superpoint_ids = np.split(global_superpoint_ids, split_indices)

    if len(superpoint_ids) == len(points):
        log.info('Superpoints created successfully.')
    else:
        log.error('Superpoints could not be created.')
        return

    # Save superpoints
    log.info(f'Saving superpoints to {directory}')
    for i, superpoints in enumerate(superpoint_ids):
        save_path = os.path.join(directory, f"{i:06d}.npy")
        np.save(save_path, superpoints.flatten())


def transform_points(scan, points: list, poses: list) -> list:
    transformed_points = []
    for p, pose in tqdm(zip(points, poses), total=len(points)):
        scan.open_scan(p)

        hom_points = np.hstack((scan.points, np.ones((scan.points.shape[0], 1))))
        transformed_points.append(np.matmul(hom_points, pose.T)[:, :3])

    return transformed_points


def down_sample_points(pcd, voxel_size: float = 0.2, std_ratio: float = 1.15,
                       nb_points: int = 20):
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Remove outliers
    _, indices = down_pcd.remove_radius_outlier(nb_points=nb_points, radius=0.5)
    down_pcd = down_pcd.select_by_index(indices)

    # Remove statistical outliers
    _, indices = down_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=std_ratio)
    down_pcd = down_pcd.select_by_index(indices)
    return down_pcd


def get_features(pcd) -> np.ndarray:
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=50))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

    # Create features vector for each point (x, y, z, nx, ny, nz)
    features = np.zeros((len(pcd.points), 6))
    features[:, :3] = np.asarray(pcd.points)
    features[:, 3:6] = np.asarray(pcd.normals)
    return features


def extend_clusters(cloud: np.ndarray, down_sampled_cloud: np.ndarray, cluster_indices: np.ndarray,
                    dist_threshold: float = 1) -> np.ndarray:
    # Create KDTree for down sampled cloud
    tree = KDTree(down_sampled_cloud)

    # Find nearest neighbors for each point in the original cloud
    distances, indices = tree.query(cloud, k=1)

    # If the distance is larger than dist_threshold, the point is an outlier do not assign a cluster
    indices[distances > dist_threshold] = -1

    # Assign cluster indices to original points
    cluster_indices = cluster_indices[indices]

    return cluster_indices
