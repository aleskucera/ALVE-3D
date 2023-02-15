import os
import sys
import sys
import logging

import scipy
import numpy as np
import open3d as o3d

from jakteristics import compute_features, FEATURE_NAMES
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from numpy.lib.recfunctions import structured_to_unstructured

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.kitti360.ply import read_ply

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt

k_nn1 = 8
k_nn2 = 40

print(FEATURE_NAMES)


def map_color(data, color_map='viridis', data_range=(0, 19), bad=(0,)):
    cmap = plt.get_cmap(color_map)
    cmap.set_bad(color='black')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # # Normalize data
    data = (data - data_range[0]) / (data_range[1] - data_range[0])
    for b in bad:
        data = np.ma.masked_where(data == b, data)

    # Map data to color
    color = m.to_rgba(data)
    return color


def visualize_geometric_features():
    sequence = 0
    window_num = 2
    seq_name = f'2013_05_28_drive_{sequence:04d}_sync'
    windows_path = os.path.join('/home/ales/Datasets/KITTI-360', 'data_3d_semantics', 'train', seq_name, 'static')
    windows = [os.path.join(windows_path, file) for file in os.listdir(windows_path) if file.endswith('.ply')]
    windows.sort()
    window = windows[window_num]

    window = read_ply(window)
    points = structured_to_unstructured(window[['x', 'y', 'z']])
    colors = structured_to_unstructured(window[['red', 'green', 'blue']]) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Filter points by voxel size
    voxel_size = 0.1
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Calculate geometric features
    geo_features = compute_features(points.astype(np.double), search_radius=0.5, max_k_neighbors=5000)

    # Anisotropy, planarity, linearity, PCA2 * 2, sphericity, verticality, surface_variation
    print(f'Shape: {geo_features.shape}')
    index = FEATURE_NAMES.index('PCA1')
    feature = geo_features[:, index]

    # Filter nan values
    feature[np.isnan(feature)] = -1

    # Visualize feature
    print(f'Max: {np.max(feature)}, Min: {np.min(feature)}')
    feature_colors = map_color(feature, color_map='viridis', data_range=(0, np.max(feature)), bad=(-1,))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(feature_colors[:, :3])
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    visualize_geometric_features()
