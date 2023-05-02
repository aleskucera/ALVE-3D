from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree
from jakteristics import compute_features


def count_colorgrad(rgb, nhoods):
    n = len(nhoods)
    out_arr = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        cur_rgb = rgb[idx]
        neigh_rgb = rgb[nhoods[idx]]
        diff = np.mean(np.abs(neigh_rgb - cur_rgb))
        out_arr[idx] = diff
    return out_arr


def compute_color_discontinuity(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    tree = KDTree(points, leaf_size=60)
    nhoods = tree.query(points, k=50, return_distance=False)
    colorgrad = count_colorgrad(colors, nhoods)
    colorgrad[colorgrad > 0.1] = 0.1
    return colorgrad


def compute_surface_variation(points: np.ndarray):
    surface_variation = compute_features(points.astype(np.double), search_radius=2,
                                         max_k_neighbors=1000, feature_names=['surface_variation'])
    surface_variation[np.isnan(surface_variation)] = -1
    return surface_variation[..., 0]
