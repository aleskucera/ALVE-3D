import os
import logging

import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mpl_toolkits.axes_grid1 import ImageGrid

from .ply import read_kitti360_ply
from src.utils.map import map_labels, map_colors
from src.utils.cloud import transform_points, downsample_cloud, nearest_neighbors, nearest_neighbors_2, \
    connected_label_components, nn_graph, visualize_cloud, visualize_cloud_values
from src.laserscan.project import project_scan

log = logging.getLogger(__name__)


class KITTI360Converter:
    def __init__(self, cfg: DictConfig):

        # ----------------- KITTI-360 structure attributes -----------------
        self.cfg = cfg
        self.sequence = cfg.conversion.sequence
        self.seq_name = f'2013_05_28_drive_{self.sequence:04d}_sync'

        self.velodyne_path = os.path.join(cfg.ds.path, 'data_3d_raw', self.seq_name, 'velodyne_points', 'data')
        self.semantics_path = os.path.join(cfg.ds.path, 'data_3d_semantics')

        self.train_windows_path = os.path.join(self.semantics_path, 'train', '2013_05_28_drive_train.txt')
        self.val_windows_path = os.path.join(self.semantics_path, 'train', '2013_05_28_drive_val.txt')

        # Transformations from camera to world frame file
        self.poses_path = os.path.join(cfg.ds.path, 'data_poses', self.seq_name, 'cam0_to_world.txt')

        # Transformation from velodyne to camera frame file
        self.calib_path = os.path.join(cfg.ds.path, 'calibration', 'calib_cam_to_velo.txt')

        # Sequence windows (global clouds)
        static_windows_path = os.path.join(self.semantics_path, 'train', self.seq_name, 'static')
        dynamic_windows_path = os.path.join(self.semantics_path, 'train', self.seq_name, 'dynamic')

        # ----------------- Sequence windows -----------------
        self.static_windows = [os.path.join(static_windows_path, f) for f in os.listdir(static_windows_path)]
        self.dynamic_windows = [os.path.join(dynamic_windows_path, f) for f in os.listdir(dynamic_windows_path)]

        self.static_windows.sort()
        self.dynamic_windows.sort()

        assert len(self.static_windows) == len(
            self.dynamic_windows), 'Number of static and dynamic windows must be equal'

        self.window_ranges = get_disjoint_ranges(self.static_windows)
        self.cloud_names = names_from_ranges(self.window_ranges)
        self.dataset_indices, self.train_samples, self.val_samples = self.get_splits(self.window_ranges,
                                                                                     self.train_windows_path,
                                                                                     self.val_windows_path)

        # ----------------- Conversion attributes -----------------
        self.static_threshold = cfg.conversion.static_threshold
        self.dynamic_threshold = cfg.conversion.dynamic_threshold

        # Sequence info
        self.num_scans = len(os.listdir(self.velodyne_path))
        self.num_windows = len(self.static_windows)

        self.semantic = None
        self.instances = None

        # ----------------- Transformations -----------------

        self.T_cam_to_velo = np.concatenate([np.loadtxt(self.calib_path).reshape(3, 4), [[0, 0, 0, 1]]], axis=0)
        self.T_velo_to_cam = np.linalg.inv(self.T_cam_to_velo)
        self.poses = read_poses(self.poses_path, self.T_velo_to_cam)

        # ----------------- Visualization attributes -----------------

        # Point clouds for visualization
        if o3d is not None:
            self.scan = o3d.geometry.PointCloud()
            self.static_window = o3d.geometry.PointCloud()
            self.dynamic_window = o3d.geometry.PointCloud()

        self.semantic_color = None
        self.instance_color = None

        # Visualization parameters
        self.scan_num = 0
        self.window_num = 0

        self.visualization_step = cfg.conversion.visualization_step

        # Color map for instance labels
        self.cmap = cm.get_cmap('Set1')
        self.cmap_length = 9

        self.key_callbacks = {
            ord(']'): self.prev_window,
            ord(']'): self.next_window,
            ord('N'): self.next_scan,
            ord('B'): self.prev_scan,
            ord('Q'): self.quit,
        }

        self.k_nn_adj = 5
        self.k_nn_local = 20

    def convert(self):
        """Convert KITTI-360 dataset to SemanticKITTI format. This function will create a sequence directory in the
        KITTI-360 dataset directory.

        The sequence directory will contain the following files:
            - velodyne: Velodyne scans
            - labels: SemanticKITTI labels
            - info.npz: Sequence info (poses, train and val samples)
        """

        # Create output directories
        sequence_path = os.path.join(self.cfg.ds.path, 'sequences', f'{self.sequence:02d}')

        labels_dir = os.path.join(sequence_path, 'labels')
        velodyne_dir = os.path.join(sequence_path, 'velodyne')

        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(velodyne_dir, exist_ok=True)

        global_clouds_dir = os.path.join(sequence_path, 'global_clouds')
        assert os.path.exists(global_clouds_dir), f'Global clouds directory {global_clouds_dir} does not exist'
        global_clouds = [os.path.join(global_clouds_dir, f) for f in os.listdir(global_clouds_dir)]
        global_clouds.sort()

        cloud_map = []

        for i, files in enumerate(zip(self.static_windows, self.dynamic_windows, global_clouds)):
            log.info(f'Converting window {i + 1}/{self.num_windows}')
            static_file, dynamic_file, global_cloud_file = files

            static_points, static_colors, semantic, _ = read_kitti360_ply(static_file)
            dynamic_points, _, _, _ = read_kitti360_ply(dynamic_file)

            with h5py.File(global_cloud_file, 'r') as f:
                global_points = np.asarray(f['points'])

            # For each scan in the window, find the points that belong to it and write them to a files
            log.info(f'Window range: {self.window_ranges[i]}')
            start, end = self.window_ranges[i]
            for j in tqdm(range(start, end + 1)):
                scan = read_scan(self.velodyne_path, j)
                scan_points = scan[:, :3]
                scan_remissions = scan[:, 3][:, np.newaxis]

                cloud_map.append(i)

                # Transform scan to the current pose
                transformed_scan_points = transform_points(scan_points, self.poses[j])

                # Find neighbors in the dynamic window and remove dynamic points
                if len(dynamic_points) > 0:
                    dists, indices = nearest_neighbors_2(dynamic_points, transformed_scan_points, k_nn=1)
                    mask = np.logical_and(dists >= 0, dists <= self.dynamic_threshold)
                    transformed_scan_points = transformed_scan_points[~mask]
                    scan_remissions = scan_remissions[~mask]
                    scan_points = scan_points[~mask]

                # Find neighbours in the static window and assign their color
                dists, indices = nearest_neighbors_2(static_points, transformed_scan_points, k_nn=1)
                mask = np.logical_and(dists >= 0, dists <= self.static_threshold)
                colors = static_colors[indices[mask]].astype(np.float32)
                semantics = semantic[indices[mask]].astype(np.uint8)

                # Find neighbours in the global cloud and assign their index
                dists, global_indices = nearest_neighbors_2(global_points, transformed_scan_points, k_nn=1)

                # Write the scan to a file
                with h5py.File(os.path.join(velodyne_dir, f'{j:06d}.h5'), 'w') as f:
                    f.create_dataset('colors', data=colors, dtype=np.float32)
                    f.create_dataset('points', data=scan_points[mask], dtype=np.float32)
                    f.create_dataset('remissions', data=scan_remissions[mask], dtype=np.float32)

                # Write the labels to a file
                with h5py.File(os.path.join(labels_dir, f'{j:06d}.h5'), 'w') as f:
                    f.create_dataset('labels', data=semantics, dtype=np.uint8)
                    f.create_dataset('voxel_map', data=global_indices, dtype=np.uint32)
                    f.create_dataset('label_mask', data=np.zeros_like(semantics, dtype=np.bool), dtype=np.bool)

        # Write the sequence info to a file
        with h5py.File(os.path.join(sequence_path, 'info.h5'), 'w') as f:
            f.create_dataset('val', data=self.val_samples)
            f.create_dataset('train', data=self.train_samples)
            f.create_dataset('poses', data=self.poses[self.dataset_indices])
            f.create_dataset('cloud_map', data=np.array(cloud_map), dtype=np.uint32)
            f.create_dataset('selection_mask', data=np.ones(len(self.train_samples), dtype=np.bool))

    def create_global_clouds(self):
        sequence_path = os.path.join(self.cfg.ds.path, 'sequences', f'{self.sequence:02d}')
        global_cloud_dir = os.path.join(sequence_path, 'global_clouds')
        os.makedirs(global_cloud_dir, exist_ok=True)

        for window_file, name in tqdm(zip(self.static_windows, self.cloud_names)):
            output_file = h5py.File(os.path.join(global_cloud_dir, f'{name}.h5'), 'w')

            # Read static window and downsample
            points, colors, labels, _ = read_kitti360_ply(window_file)
            points, colors, labels = downsample_cloud(points, colors, labels, 0.2)

            # Compute graph edges
            edge_sources, edge_targets, distances = nn_graph(points, self.k_nn_adj)

            # Compute local neighbors
            local_neighbors = nearest_neighbors(points, self.k_nn_local)

            # Computes object in point cloud and transition edges
            objects = connected_label_components(labels, edge_sources, edge_targets)
            edge_transitions = objects[edge_sources] != objects[edge_targets]

            output_file.create_dataset('points', data=points, dtype='float32')
            output_file.create_dataset('colors', data=colors, dtype='float32')
            output_file.create_dataset('labels', data=labels, dtype='uint8')
            output_file.create_dataset('objects', data=objects, dtype='uint32')

            output_file.create_dataset('edge_sources', data=edge_sources, dtype='uint32')
            output_file.create_dataset('edge_targets', data=edge_targets, dtype='uint32')
            output_file.create_dataset('edge_transitions', data=edge_transitions, dtype='uint8')
            output_file.create_dataset('local_neighbors', data=local_neighbors, dtype='uint32')

    def update_window(self):
        static_points, static_colors, self.semantic, _ = read_kitti360_ply(self.static_windows[self.window_num])

        self.semantic = map_labels(self.semantic, self.cfg.ds.learning_map).flatten()
        self.semantic_color = map_colors(self.semantic, self.cfg.ds.color_map_train)

        dynamic_points, _, _, _ = read_kitti360_ply(self.dynamic_windows[self.window_num])
        dynamic_colors = np.ones_like(dynamic_points) * [0, 0, 1]

        self.static_window.points = o3d.utility.Vector3dVector(static_points)
        self.static_window.colors = o3d.utility.Vector3dVector(static_colors)

        self.dynamic_window.points = o3d.utility.Vector3dVector(dynamic_points)
        self.dynamic_window.colors = o3d.utility.Vector3dVector(dynamic_colors)

        self.scan_num = self.window_ranges[self.window_num][0]
        self.update_scan()

    def update_scan(self):

        # Read scan
        scan = read_scan(self.velodyne_path, self.scan_num)
        scan_points = scan[:, :3]

        # Transform scan to world coordinates
        transformed_scan_points = transform_points(scan_points, self.poses[self.scan_num])

        # Find neighbours in the static window
        dists, indices = nearest_neighbors_2(self.static_window.points, transformed_scan_points, k_nn=1)
        mask = np.logical_and(dists >= 0, dists <= self.static_threshold)

        # Extract RGB values from the static window
        rgb = np.array(self.static_window.colors)[indices[mask]]

        # Color of the scan in world coordinates
        scan_colors = np.ones_like(scan_points) * [1, 0, 0]
        scan_colors[mask] = [0, 1, 0]

        # Get point cloud labels
        semantics = np.array(self.semantic_color)[indices[mask]]

        # Project the scan to the camera
        projection = project_scan(scan_points, 64, 1024, 3, -25.0)
        proj_mask = projection['mask']

        # Project the filtered scan to the camera
        filtered_projection = project_scan(scan_points[mask], 64, 1024, 3, -25.0)
        filtered_proj_mask = filtered_projection['mask']
        filtered_proj_indices = filtered_projection['idx'][filtered_proj_mask]

        # Project color, semantic and instance labels to the camera
        proj_color = np.zeros((64, 1024, 3), dtype=np.float32)
        proj_semantics = np.zeros((64, 1024, 3), dtype=np.float32)
        proj_instances = np.zeros((64, 1024, 3), dtype=np.float32)

        proj_color[filtered_proj_mask] = rgb[filtered_proj_indices]
        proj_semantics[filtered_proj_mask] = semantics[filtered_proj_indices]

        # Visualize the projection
        fig = plt.figure(figsize=(11, 4), dpi=150)
        grid = ImageGrid(fig, 111, nrows_ncols=(5, 1), axes_pad=0.4)

        images = [proj_mask, filtered_proj_mask, proj_color, proj_semantics, proj_instances]
        titles = ['Projection Mask', 'Filtered Projection Mask', 'RGB Color', 'Semantic Labels', 'Instance Labels']

        for ax, image, title in zip(grid, images, titles):
            ax.set_title(title)
            ax.imshow(image, aspect='auto')
            ax.axis('off')

        plt.show()

        self.scan.points = o3d.utility.Vector3dVector(transformed_scan_points)
        self.scan.colors = o3d.utility.Vector3dVector(scan_colors)

    def get_splits(self, window_ranges: list[tuple[int, int]], train_path: str, val_path: str) -> tuple:

        dataset_indices = []
        val_indices = []
        train_indices = []

        train_ranges = [get_window_range(path) for path in read_txt(train_path, self.seq_name)]
        val_ranges = [get_window_range(path) for path in read_txt(val_path, self.seq_name)]

        for window in window_ranges:
            dataset_indices += list(range(window[0], window[1] + 1))
            for train_range in train_ranges:
                if train_range[0] == window[0]:
                    train_indices += list(range(window[0], window[1] + 1))
            for val_range in val_ranges:
                if val_range[0] == window[0]:
                    val_indices += list(range(window[0], window[1] + 1))

        train_samples = np.searchsorted(dataset_indices, train_indices)
        val_samples = np.searchsorted(dataset_indices, val_indices)
        return dataset_indices, train_samples, val_samples

    def next_window(self, vis):
        self.window_num += 1
        self.update_window()
        vis.update_geometry(self.static_window)
        vis.update_geometry(self.dynamic_window)
        vis.update_geometry(self.scan)
        vis.reset_view_point(True)
        vis.update_renderer()
        return False

    def prev_window(self, vis):
        self.window_num -= self.visualization_step
        self.update_window()
        vis.update_geometry(self.static_window)
        vis.update_geometry(self.dynamic_window)
        vis.update_geometry(self.scan)
        vis.reset_view_point(True)
        vis.update_renderer()
        return False

    def next_scan(self, vis):
        self.scan_num += self.visualization_step
        self.update_scan()
        vis.update_geometry(self.scan)
        vis.update_renderer()
        return False

    def prev_scan(self, vis):
        self.scan_num -= self.visualization_step
        self.update_scan()
        vis.update_geometry(self.scan)
        vis.update_renderer()
        return False

    @staticmethod
    def quit(vis):
        vis.destroy_window()
        return True

    def visualize(self):
        log.info('Visualizing the KITTI-360 conversion')
        print('\nControls:')
        print('  - Press "b" to go to the next scan')
        print('  - Press "p" to go to the previous scan')
        print('  - Press "]" to go to the next window')
        print('  - Press "[" to go to the previous window')
        self.update_window()
        self.update_scan()
        o3d.visualization.draw_geometries_with_key_callbacks(
            [self.static_window, self.scan],
            self.key_callbacks)


def read_poses(poses_path: str, T_velo2cam: np.ndarray) -> np.ndarray:
    """Read poses from poses.txt file. The poses are transformations from the velodyne coordinate
    system to the world coordinate system.
    :return: array of poses (Nx4x4), where N is the number of velodyne scans
    """

    # Load poses. Some poses are missing, because the camera was not moving.
    compressed_poses = np.loadtxt(poses_path, dtype=np.float32)
    frames = compressed_poses[:, 0].astype(np.int32)
    lidar_poses = compressed_poses[:, 1:].reshape(-1, 4, 4)

    # Create a full list of poses (with missing poses with value of the last known pose)
    sequence_length = np.max(frames) + 1
    poses = np.zeros((sequence_length, 4, 4), dtype=np.float32)

    last_valid_pose = lidar_poses[0]
    for i in range(sequence_length):
        if i in frames:
            last_valid_pose = lidar_poses[frames == i] @ T_velo2cam
        poses[i] = last_valid_pose
    return poses


def read_txt(file: str, sequence_name: str):
    """ Read file with one value per line. """
    with open(file, 'r') as f:
        lines = f.readlines()
    return [os.path.basename(line.strip()) for line in lines if sequence_name in line]


def get_window_range(path: str):
    """ Parse string of form '/path/to/file/start_end.ply' and return range of integers. """
    file_name = os.path.basename(path)
    file_name = os.path.splitext(file_name)[0]
    split_file_name = file_name.split('_')

    assert len(split_file_name) == 2, f'Invalid file name: {file_name}'
    start, end = split_file_name
    return int(start), int(end)


def read_scan(velodyne_path: str, i: int):
    """ Read velodyne scan from binary file. """
    file = os.path.join(velodyne_path, f'{i:010d}.bin')
    scan = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    return scan


def get_list(ranges: list[tuple[int, int]]) -> list[int]:
    """ Create list of sorted values without duplicates. """
    values = []
    for start, end in ranges:
        values.extend(range(start, end))
    return sorted(set(values))


def split_indices(ranges: list[tuple[int, int]], all_indices: list) -> np.ndarray:
    """ Create list of sorted indices without duplicates. """
    indices = get_list(ranges)
    return np.searchsorted(all_indices, indices)


def get_disjoint_ranges(paths: list[str]) -> list[tuple[int, int]]:
    """ Create list of disjoint ranges. """
    paths.sort()
    ranges = [get_window_range(path) for path in paths]
    disjoint_ranges = []
    for i, (start, end) in enumerate(ranges):
        if i == len(ranges) - 1:
            disjoint_ranges.append((start, end))
        else:
            next_start, _ = ranges[i + 1]
            if next_start < end:
                end = next_start - 1
            disjoint_ranges.append((start, end))
    return disjoint_ranges


def names_from_ranges(window_ranges: list[tuple[int, int]]) -> list[str]:
    """ Create list of window names. """
    return [f'{start:06d}_{end:06d}' for start, end in window_ranges]
