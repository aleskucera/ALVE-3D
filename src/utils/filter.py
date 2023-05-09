import numpy as np
import open3d as o3d


def filter_scan(points: np.ndarray, filter_type: str) -> np.ndarray:
    if filter_type == 'distance':
        return distant_points(points, 30)
    elif filter_type == 'radius':
        return radius_outliers(points, 10, 0.5)
    elif filter_type == 'statistical':
        return statistical_outliers(points, 10, 1.8)
    else:
        raise ValueError(f'Invalid scan filter: {filter_type}')


def distant_points(points: np.ndarray, distance: float) -> np.ndarray:
    mask = np.linalg.norm(points, axis=1) > distance
    return np.where(mask)[0]


def radius_outliers(points: np.ndarray, nb_points: int, radius: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    _, indices = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    outliers_indices = np.setdiff1d(np.arange(points.shape[0]), indices)
    return outliers_indices


def statistical_outliers(points: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    _, indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    outliers_indices = np.setdiff1d(np.arange(points.shape[0]), indices)
    return outliers_indices
