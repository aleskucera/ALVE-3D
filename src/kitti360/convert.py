import os
import logging

import h5py
import numpy as np
from tqdm import tqdm
from .ply import read_kitti360_ply
from src.utils import transform_points, downsample_cloud, nearest_neighbors, \
    nearest_neighbors_2, connected_label_components, nn_graph, map_labels
from .utils import read_kitti360_scan

log = logging.getLogger(__name__)

K_NN_ADJ = 5
K_NN_LOCAL = 20
VOXEL_SIZE = 0.2
STATIC_THRESHOLD = 0.2
DYNAMIC_THRESHOLD = 0.2


def convert_sequence(sequence_path: str,
                     velodyne_dir: str,
                     poses: np.ndarray,
                     train_samples: np.ndarray,
                     val_samples: np.ndarray,
                     train_clouds: np.ndarray,
                     val_clouds: np.ndarray,
                     static_windows: list[str],
                     dynamic_windows: list[str],
                     window_ranges: list[tuple[int, int]],
                     label_map: dict[int, int],
                     ignore_index: int):
    # Velodyne directory
    scans_dir = os.path.join(sequence_path, 'velodyne')
    os.makedirs(scans_dir, exist_ok=True)

    # Clouds directory
    clouds_dir = os.path.join(sequence_path, 'voxel_clouds')
    os.makedirs(clouds_dir, exist_ok=True)

    clouds = np.sort(np.concatenate([train_clouds, val_clouds]))
    clouds = [os.path.join(clouds_dir, cloud) for cloud in clouds]

    val_samples, train_samples = val_samples.astype('S'), train_samples.astype('S')
    val_clouds, train_clouds = val_clouds.astype('S'), train_clouds.astype('S')

    # Write the sequence info to a file
    with h5py.File(os.path.join(sequence_path, 'info.h5'), 'w') as f:
        f.create_dataset('val', data=val_samples)
        f.create_dataset('train', data=train_samples)
        f.create_dataset('val_clouds', data=val_clouds)
        f.create_dataset('train_clouds', data=train_clouds)

    for data in zip(static_windows, dynamic_windows, clouds, window_ranges):
        static_file, dynamic_file, cloud_file, window_range = data
        dynamic_points, _, _, _ = read_kitti360_ply(dynamic_file)
        static_points, static_colors, static_labels, _ = read_kitti360_ply(static_file)

        # Filter out the points that will be mapped to ignore_index
        learn_labels = map_labels(static_labels.flatten(), label_map)
        mask = learn_labels != ignore_index
        static_points = static_points[mask]
        static_colors = static_colors[mask]
        static_labels = static_labels[mask]

        voxel_points, voxel_colors, voxel_labels = downsample_cloud(static_points, static_colors,
                                                                    static_labels, VOXEL_SIZE)

        voxel_mask = np.zeros(len(voxel_points), dtype=np.bool)

        start, end = window_range
        for j in tqdm(range(start, end + 1), desc=f'Creating scans {start} - {end}'):
            scan = read_kitti360_scan(velodyne_dir, j)
            scan_points, scan_remissions = scan[:, :3], scan[:, 3]

            # Transform scan to the current pose
            transformed_scan_points = transform_points(scan_points, poses[j])

            # Find neighbors in the dynamic window and remove dynamic points
            if len(dynamic_points) > 0:
                dists, indices = nearest_neighbors_2(dynamic_points, transformed_scan_points, k_nn=1)
                mask = np.logical_and(dists >= 0, dists <= DYNAMIC_THRESHOLD)
                transformed_scan_points = transformed_scan_points[~mask]
                scan_remissions = scan_remissions[~mask]
                scan_points = scan_points[~mask]

            # Find neighbours in the static window and assign their color
            dists, indices = nearest_neighbors_2(static_points, transformed_scan_points, k_nn=1)
            mask = np.logical_and(dists >= 0, dists <= STATIC_THRESHOLD)
            colors = static_colors[indices[mask]].astype(np.float32)
            transformed_scan_points = transformed_scan_points[mask]
            scan_remissions = scan_remissions[mask]
            scan_points = scan_points[mask]

            # Find neighbours in the voxel cloud and assign their label and index
            dists, voxel_indices = nearest_neighbors_2(voxel_points, transformed_scan_points, k_nn=1)
            labels = voxel_labels[voxel_indices].astype(np.uint8)
            voxel_mask[voxel_indices] = True

            # Write the scan to a file
            with h5py.File(os.path.join(scans_dir, f'{j:06d}.h5'), 'w') as f:
                f.create_dataset('pose', data=poses[j], dtype=np.float32)
                f.create_dataset('colors', data=colors, dtype=np.float32)
                f.create_dataset('points', data=scan_points, dtype=np.float32)
                f.create_dataset('remissions', data=scan_remissions, dtype=np.float32)

                f.create_dataset('labels', data=labels, dtype=np.uint8)
                f.create_dataset('voxel_map', data=voxel_indices.astype(np.uint32), dtype=np.uint32)

        filter_map = np.cumsum(voxel_mask) - 1
        for j in tqdm(range(start, end + 1), desc=f'Changing voxel maps {start} - {end}'):
            with h5py.File(os.path.join(scans_dir, f'{j:06d}.h5'), 'r+') as f:
                voxel_map = np.asarray(f['voxel_map'])
                f['voxel_map'][:] = filter_map[voxel_map]

        # Filter unused voxels
        voxel_points = voxel_points[voxel_mask]
        voxel_colors = voxel_colors[voxel_mask]
        voxel_labels = voxel_labels[voxel_mask]

        # Compute graph properties
        local_neighbors, _ = nearest_neighbors(voxel_points, K_NN_LOCAL)
        edge_sources, edge_targets, distances = nn_graph(voxel_points, K_NN_ADJ)

        # Computes object in point cloud based on semantic labels
        objects = connected_label_components(voxel_labels, edge_sources, edge_targets)

        with h5py.File(cloud_file, 'w') as f:
            f.create_dataset('points', data=voxel_points, dtype='float32')
            f.create_dataset('colors', data=voxel_colors, dtype='float32')

            f.create_dataset('objects', data=objects.flatten(), dtype='uint32')
            f.create_dataset('labels', data=voxel_labels.flatten(), dtype='uint8')

            f.create_dataset('local_neighbors', data=local_neighbors, dtype='uint32')
            f.create_dataset('edge_sources', data=edge_sources.flatten(), dtype='uint32')
            f.create_dataset('edge_targets', data=edge_targets.flatten(), dtype='uint32')
