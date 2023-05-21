import numpy as np
import open3d as o3d

from src.utils.cloud import nearest_neighbors


def filter_scan(points: np.ndarray, filter_type: str) -> np.ndarray:
    if filter_type == 'Distance':
        return distant_points(points, 30)
    elif filter_type == 'Radius':
        return radius_outliers(points, 20, 0.1)
    else:
        raise ValueError(f'Invalid scan filter: {filter_type}')


def distant_points(points: np.ndarray, distance: float) -> np.ndarray:
    mask = np.linalg.norm(points, axis=1) > distance
    return np.where(mask)[0]


def radius_outliers(points: np.ndarray, nb_points: int, radius: float) -> np.ndarray:
    distances, neighbors = nearest_neighbors(points, k_nn=nb_points)
    mask = np.any(distances > radius, axis=1)
    return np.where(mask)[0]
