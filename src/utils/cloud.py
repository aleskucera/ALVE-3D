import time
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor

from src.ply_c import libply_c
from src.utils.map import colorize, colorize_instances


def visualize_cloud(points: np.ndarray, colors: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def visualize_cloud_values(points: np.ndarray, values: np.ndarray, random_colors: bool = False):
    if random_colors:
        colors = colorize_instances(values, max_inst_id=np.max(values) + 1)
    else:
        colors = colorize(values, color_map='viridis', data_range=(np.min(values), np.max(values)))
    visualize_cloud(points, colors)


def downsample_cloud(points: np.ndarray, colors: np.ndarray, labels: np.ndarray, voxel_size=0.1):
    device = o3d.core.Device("CPU:0")
    pcd = o3d.t.geometry.PointCloud(device)

    pcd.point.positions = o3d.core.Tensor(points, o3d.core.float32, device)
    pcd.point.colors = o3d.core.Tensor(colors, o3d.core.float32, device)
    pcd.point.labels = o3d.core.Tensor(labels, o3d.core.uint32, device)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd.point.positions.numpy(), pcd.point.colors.numpy(), pcd.point.labels.numpy()


def nearest_neighbors(points: np.ndarray, k_nn: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='kd_tree').fit(points)
    return np.array(nn.kneighbors(points))[..., 1:].astype(np.uint32)


def nearest_neighbors_2(points_1: np.ndarray, points_2: np.ndarray, k_nn: int) -> tuple:
    nn = NearestNeighbors(n_neighbors=k_nn, algorithm='kd_tree').fit(points_1)
    distances, neighbors = nn.kneighbors(points_2)
    if k_nn == 1:
        return distances.flatten(), neighbors.flatten()
    return distances, neighbors


def nn_graph(points: np.ndarray, k_nn: int):
    n_ver = points.shape[0]

    distances, neighbors = nearest_neighbors(points, k_nn)

    indices = np.arange(0, n_ver)[..., np.newaxis]
    source = np.zeros((n_ver, k_nn)) + indices

    edge_sources = source.flatten().astype(np.uint32)
    edge_targets = neighbors.flatten().astype(np.uint32)
    distances = distances.flatten().astype(np.float32)
    return edge_sources, edge_targets, distances


def connected_label_components(labels: np.ndarray, edg_source: np.ndarray, edg_target: np.ndarray):
    label_transition = labels[edg_source] != labels[edg_target]
    _, components = libply_c.connected_comp(len(labels), edg_source.astype('uint32'),
                                            edg_target.astype('uint32'),
                                            (label_transition == 0).astype('uint8'), 0)
    return components


def compute_elevation(points: np.ndarray, plane_threshold=5, min_plane_points=1000):
    z = points[:, 2]
    xy = points[:, :2]
    plane = (z - z.min() < plane_threshold).nonzero()[0]
    if len(plane) > min_plane_points:
        reg = RANSACRegressor(random_state=0).fit(xy[plane], z[plane])
        elevation = z - reg.predict(xy)
    else:
        elevation = np.zeros(len(points))
    return elevation


def normalize_xy(xy: np.ndarray):
    max_xy, min_xy = np.max(xy, axis=0, keepdims=True), np.min(xy, axis=0, keepdims=True)
    return (xy - min_xy) / (max_xy - min_xy)


def transform_points(points: np.ndarray, transform: np.ndarray):
    points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return np.matmul(points, transform.T)[:, :3]
