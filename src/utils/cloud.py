import random

import numpy as np
import open3d as o3d
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R

import wandb
from src.ply_c import libply_c
from src.utils.map import colorize_values, colorize_instances


def visualize_cloud(points: np.ndarray, colors: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def visualize_cloud_values(points: np.ndarray, values: np.ndarray, random_colors: bool = False):
    if random_colors:
        colors = colorize_instances(values, max_inst_id=np.max(values) + 1)
    else:
        colors = colorize_values(values, color_map='viridis', data_range=(np.min(values), np.max(values)))
    visualize_cloud(points, colors)


def downsample_cloud(points: np.ndarray, colors: np.ndarray = None, labels: np.ndarray = None,
                     voxel_size: float = 0.1) -> tuple:
    device = o3d.core.Device("CPU:0")
    pcd = o3d.t.geometry.PointCloud(device)

    pcd.point.positions = o3d.core.Tensor(points, o3d.core.float32, device)
    if colors is not None:
        pcd.point.colors = o3d.core.Tensor(colors, o3d.core.float32, device)
    if labels is not None:
        pcd.point.labels = o3d.core.Tensor(labels, o3d.core.uint32, device)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if labels is not None and colors is not None:
        return pcd.point.positions.numpy(), pcd.point.colors.numpy(), pcd.point.labels.numpy()
    elif labels is not None:
        return pcd.point.positions.numpy(), pcd.point.labels.numpy()
    elif colors is not None:
        return pcd.point.positions.numpy(), pcd.point.colors.numpy()
    else:
        return pcd.point.positions.numpy()


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


def visualize_global_cloud(points: iter, colors: iter, poses: iter, voxel_size: float = 0.2, log: bool = False) -> None:
    cloud, cloud_color = [], []
    for p, c, pos in tqdm(zip(points, colors, poses), total=len(points), desc='Creating global cloud'):
        transformed_points = transform_points(p, pos)

        cloud.append(transformed_points)
        cloud_color.append(c)

    cloud = np.concatenate(cloud, axis=0)
    color = np.concatenate(cloud_color, axis=0)

    # Down sample global cloud with voxel grid
    cloud, color = downsample_cloud(cloud, color, voxel_size=voxel_size)

    if log:
        wandb.log({'global_cloud': [wandb.Object3D(np.concatenate([cloud, color * 255], axis=1))]})
    else:
        visualize_cloud(cloud, color)


def calculate_radial_distances(points: np.ndarray, center: np.ndarray = None):
    if center is None:
        center = np.zeros(3)
    return np.linalg.norm(points - center, axis=1)


def augment_points(points: np.ndarray, translation_prob: float = 0.5, rotation_prob: float = 0.5,
                   flip_prob: float = 0.5, drop_prob: float = 0.5):
    if random.random() < flip_prob:
        points[:, 0] *= -1

    # Translate points
    if random.random() < translation_prob:
        points[:, 0] += random.uniform(-5, 5)
        points[:, 1] += random.uniform(-3, 3)
        points[:, 2] += random.uniform(-1, 0)

    # Rotate points
    if random.random() < rotation_prob:
        deg = random.uniform(-180, 180)
        rot = R.from_euler('z', deg, degrees=True)
        points = rot.apply(points)

    # Drop points
    rng = np.random.default_rng()
    drop_mask = rng.choice([False, True], size=points.shape[0], p=[drop_prob, 1 - drop_prob])
    return points[drop_mask], drop_mask
