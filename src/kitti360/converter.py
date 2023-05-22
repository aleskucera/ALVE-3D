import os
import logging

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mpl_toolkits.axes_grid1 import ImageGrid

from .ply import read_kitti360_ply
from src.utils.project import project_points
from src.utils.map import map_labels, map_colors
from .convert import convert_sequence, STATIC_THRESHOLD
from src.utils.cloud import transform_points, nearest_neighbors_2, downsample_cloud
from .utils import get_disjoint_ranges, read_kitti360_poses, \
    read_kitti360_scan, get_window_range, read_txt

log = logging.getLogger(__name__)


class KITTI360Converter:
    """ Converter for the KITTI-360 dataset.
    The object provides the following functionalities:
        - Convert a sequence to a format for active learning experiments.
        - Visualize the conversion of a sequence.
    """

    def __init__(self, cfg: DictConfig):

        # ----------------- KITTI-360 structure attributes -----------------
        self.cfg = cfg
        self.sequence = cfg.sequence if 'sequence' in cfg else 0
        self.seq_name = f'2013_05_28_drive_{self.sequence:04d}_sync'

        self.velodyne_path = os.path.join(cfg.ds.path, 'data_3d_raw', self.seq_name, 'velodyne_points', 'data')
        self.semantics_path = os.path.join(cfg.ds.path, 'data_3d_semantics')

        self.train_windows_path = os.path.join(self.semantics_path, 'train', '2013_05_28_drive_train.txt')
        self.val_windows_path = os.path.join(self.semantics_path, 'train', '2013_05_28_drive_val.txt')

        # Transformations from camera to world frame file
        self.poses_path = os.path.join(cfg.ds.path, 'data_poses', self.seq_name, 'cam0_to_world.txt')

        # Transformation from velodyne to camera frame file
        self.calib_path = os.path.join(cfg.ds.path, 'calibration', 'calib_cam_to_velo.txt')

        static_windows_dir = os.path.join(self.semantics_path, 'train', self.seq_name, 'static')
        dynamic_windows_dir = os.path.join(self.semantics_path, 'train', self.seq_name, 'dynamic')
        self.static_windows = sorted([os.path.join(static_windows_dir, f) for f in os.listdir(static_windows_dir)])
        self.dynamic_windows = sorted([os.path.join(dynamic_windows_dir, f) for f in os.listdir(dynamic_windows_dir)])

        # Create a disjoint set of ranges for the windows
        self.window_ranges = get_disjoint_ranges(self.static_windows)

        # Get the train and validation splits with the corresponding new cloud names (without overlap)
        splits = self.get_splits(self.window_ranges, self.train_windows_path, self.val_windows_path)
        self.train_samples, self.val_samples, self.train_clouds, self.val_clouds = splits

        # Sequence info
        self.num_scans = len(os.listdir(self.velodyne_path))
        self.num_windows = len(self.static_windows)

        # ----------------- Transformations -----------------

        self.T_cam_to_velo = np.concatenate([np.loadtxt(self.calib_path).reshape(3, 4), [[0, 0, 0, 1]]], axis=0)
        self.T_velo_to_cam = np.linalg.inv(self.T_cam_to_velo)
        self.poses = read_kitti360_poses(self.poses_path, self.T_velo_to_cam)

        # ----------------- Visualization attributes -----------------

        # Point clouds for visualization
        self.scan = o3d.geometry.PointCloud()
        self.void_cloud = o3d.geometry.PointCloud()
        self.static_cloud = o3d.geometry.PointCloud()
        self.dynamic_cloud = o3d.geometry.PointCloud()

        self.void_points = None
        self.void_colors = None
        self.label_colors = None
        self.static_points = None
        self.static_colors = None
        self.dynamic_points = None
        self.dynamic_colors = None

        self.static_rgb = None
        self.scan_visible = True
        self.void_visible = True
        self.dynamic_visible = True

        # Visualization parameters
        self.scan_num = 0
        self.window_num = 0
        self.visualization_step = cfg.conversion.visualization_step
        self.key_callbacks = {
            ord('['): self.prev_window,
            ord(']'): self.next_window,
            ord('N'): self.next_scan,
            ord('B'): self.prev_scan,
            ord('D'): self.toggle_dynamic_visibility,
            ord('V'): self.toggle_void_visibility,
            ord('S'): self.toggle_scan_visibility,
            ord('C'): self.toggle_static_color,
            ord('Q'): self.quit,
        }

    def convert(self):
        sequence_path = os.path.join(self.cfg.ds.path, 'sequences', f'{self.sequence:02d}')
        convert_sequence(sequence_path=sequence_path,
                         velodyne_dir=self.velodyne_path,
                         poses=self.poses,
                         train_samples=self.train_samples,
                         val_samples=self.val_samples,
                         train_clouds=self.train_clouds,
                         val_clouds=self.val_clouds,
                         static_windows=self.static_windows,
                         dynamic_windows=self.dynamic_windows,
                         window_ranges=self.window_ranges,
                         label_map=self.cfg.ds.learning_map,
                         ignore_index=self.cfg.ds.ignore_index)

    def get_splits(self, window_ranges: list[tuple[int, int]], train_file: str, val_file: str) -> tuple:
        """ Get the train and validation splits for the dataset. Also returns the cloud names.

        :param window_ranges: List of tuples containing the start and end scan of each window.
        :param train_file: Path to the train split file.
        :param val_file: Path to the validation split file.
        :return: Tuple containing the train and validation splits and the cloud names.
        """

        val_clouds = np.array([], dtype=np.str_)
        val_samples = np.array([], dtype=np.str_)
        train_clouds = np.array([], dtype=np.str_)
        train_samples = np.array([], dtype=np.str_)

        val_ranges = [get_window_range(path) for path in read_txt(val_file, self.seq_name)]
        train_ranges = [get_window_range(path) for path in read_txt(train_file, self.seq_name)]

        for i, window in enumerate(window_ranges):
            cloud_name = f'{window[0]:06d}_{window[1]:06d}.h5'
            window_samples = np.array([f'{j:06d}.h5' for j in np.arange(window[0], window[1] + 1)], dtype='S')
            for train_range in train_ranges:
                if train_range[0] == window[0]:
                    train_samples = np.concatenate((train_samples, window_samples))
                    train_clouds = np.append(train_clouds, cloud_name)
            for val_range in val_ranges:
                if val_range[0] == window[0]:
                    val_samples = np.concatenate((val_samples, window_samples))
                    val_clouds = np.append(val_clouds, cloud_name)

        return train_samples, val_samples, train_clouds, val_clouds

    def update_window(self):

        # Load the static window
        static_points, static_colors, labels, _ = read_kitti360_ply(self.static_windows[self.window_num])
        static_points, static_colors, labels = downsample_cloud(static_points, static_colors, labels, 0.2)

        # Create colors for the points
        labels = map_labels(labels, self.cfg.ds.learning_map).flatten()
        label_colors = map_colors(labels, self.cfg.ds.color_map_train)

        # Filter out void points
        void_indices = np.where(labels == self.cfg.ds.ignore_index)[0]
        labeled_indices = np.where(labels != self.cfg.ds.ignore_index)[0]

        self.label_colors = label_colors[labeled_indices]
        self.static_points = static_points[labeled_indices]
        self.static_colors = static_colors[labeled_indices]

        # self.void_colors = label_colors[void_indices]
        self.void_points = static_points[void_indices]
        self.void_colors = np.full((self.void_points.shape[0], 3), [0, 0, 0], dtype=np.uint8)

        # Load the dynamic window
        self.dynamic_points, _, _, _ = read_kitti360_ply(self.dynamic_windows[self.window_num])
        self.dynamic_colors = np.full((self.dynamic_points.shape[0], 3), [0, 0, 1], dtype=np.uint8)

        self.static_rgb, self.scan_visible = True, True
        self.void_visible, self.dynamic_visible, = True, True

        self.void_cloud.points = o3d.utility.Vector3dVector(self.void_points)
        self.void_cloud.colors = o3d.utility.Vector3dVector(self.void_colors)

        self.static_cloud.points = o3d.utility.Vector3dVector(self.static_points)
        self.static_cloud.colors = o3d.utility.Vector3dVector(self.static_colors)

        self.dynamic_cloud.points = o3d.utility.Vector3dVector(self.dynamic_points)
        self.dynamic_cloud.colors = o3d.utility.Vector3dVector(self.dynamic_colors)

        # self.static_window.points = o3d.utility.Vector3dVector(static_points)
        # self.static_window.colors = o3d.utility.Vector3dVector(np.full(static_points.shape, [0.7, 0.7, 0.7]))
        #
        # self.void_points.points = o3d.utility.Vector3dVector(void_points)
        # self.void_points.colors = o3d.utility.Vector3dVector(void_colors)
        #
        # self.dynamic_window.points = o3d.utility.Vector3dVector(dynamic_points)
        # self.dynamic_window.colors = o3d.utility.Vector3dVector(dynamic_colors)

        self.scan_num = self.window_ranges[self.window_num][0]
        self.update_scan()

    def update_scan(self):

        # Read scan
        scan = read_kitti360_scan(self.velodyne_path, self.scan_num)
        self.scan_points = transform_points(scan[:, :3], self.poses[self.scan_num])

        # Find neighbours in the static window
        dists, indices = nearest_neighbors_2(self.static_points, self.scan_points, k_nn=1)
        mask = np.logical_and(dists >= 0, dists <= STATIC_THRESHOLD)
        self.scan_points += [0, 0, 0.1]

        # Extract RGB values from the static window
        # rgb = np.array(self.static_colors)[indices[mask]]

        # Color of the scan in world coordinates
        self.scan_colors = np.ones_like(self.scan_points) * [1, 0, 0]
        self.scan_colors[mask] = [0, 1, 0]
        #
        # # Get point cloud labels
        # semantics = np.array(self.label_colors)[indices[mask]]
        #
        # # Project the scan to the camera
        # projection = project_points(scan_points, 64, 1024, 3, -25.0)
        # proj_mask = projection['mask']
        #
        # # Project the filtered scan to the camera
        # filtered_projection = project_points(scan_points[mask], 64, 1024, 3, -25.0)
        # filtered_proj_mask = filtered_projection['mask']
        # filtered_proj_indices = filtered_projection['idx'][filtered_proj_mask]
        #
        # # Project color, semantic and instance labels to the camera
        # proj_color = np.zeros((64, 1024, 3), dtype=np.float32)
        # proj_semantics = np.zeros((64, 1024, 3), dtype=np.float32)
        # proj_instances = np.zeros((64, 1024, 3), dtype=np.float32)
        #
        # proj_color[filtered_proj_mask] = rgb[filtered_proj_indices]
        # proj_semantics[filtered_proj_mask] = semantics[filtered_proj_indices]
        #
        # # Visualize the projection
        # fig = plt.figure(figsize=(11, 4), dpi=150)
        # grid = ImageGrid(fig, 111, nrows_ncols=(5, 1), axes_pad=0.4)
        #
        # images = [proj_mask, filtered_proj_mask, proj_color, proj_semantics, proj_instances]
        # titles = ['Projection Mask', 'Filtered Projection Mask', 'RGB Color', 'Semantic Labels', 'Instance Labels']
        #
        # for ax, image, title in zip(grid, images, titles):
        #     ax.set_title(title)
        #     ax.imshow(image, aspect='auto')
        #     ax.axis('off')
        #
        # plt.show()

        self.scan.points = o3d.utility.Vector3dVector(self.scan_points)
        self.scan.colors = o3d.utility.Vector3dVector(self.scan_colors)

    def next_window(self, vis):
        self.window_num += 1
        self.update_window()
        vis.update_geometry(self.scan)
        vis.update_geometry(self.void_cloud)
        vis.update_geometry(self.static_cloud)
        vis.update_geometry(self.dynamic_cloud)
        vis.reset_view_point(True)
        vis.update_renderer()
        return False

    def prev_window(self, vis):
        self.window_num -= 1
        self.update_window()
        vis.update_geometry(self.scan)
        vis.update_geometry(self.void_cloud)
        vis.update_geometry(self.static_cloud)
        vis.update_geometry(self.dynamic_cloud)
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

    def toggle_static_color(self, vis):
        self.static_rgb = not self.static_rgb

        if self.static_rgb:
            self.static_cloud.colors = o3d.utility.Vector3dVector(self.static_colors)
        else:
            self.static_cloud.colors = o3d.utility.Vector3dVector(self.label_colors)

        vis.update_geometry(self.static_cloud)
        vis.update_renderer()
        return False

    def toggle_scan_visibility(self, vis):
        self.scan_visible = not self.scan_visible

        if self.scan_visible:
            self.scan.points = o3d.utility.Vector3dVector(self.scan_points)
            self.scan.colors = o3d.utility.Vector3dVector(self.scan_colors)
        else:
            self.scan.points = o3d.utility.Vector3dVector([])
            self.scan.colors = o3d.utility.Vector3dVector([])

        vis.update_geometry(self.scan)
        vis.update_renderer()
        return False

    def toggle_dynamic_visibility(self, vis):
        self.dynamic_visible = not self.dynamic_visible

        if self.dynamic_visible:
            self.dynamic_cloud.points = o3d.utility.Vector3dVector(self.dynamic_points)
            self.dynamic_cloud.colors = o3d.utility.Vector3dVector(self.dynamic_colors)
        else:
            self.dynamic_cloud.points = o3d.utility.Vector3dVector([])
            self.dynamic_cloud.colors = o3d.utility.Vector3dVector([])

        vis.update_geometry(self.dynamic_cloud)
        vis.update_renderer()
        return False

    def toggle_void_visibility(self, vis):
        self.void_visible = not self.void_visible

        if self.void_visible:
            self.void_cloud.points = o3d.utility.Vector3dVector(self.void_points)
            self.void_cloud.colors = o3d.utility.Vector3dVector(self.void_colors)
        else:
            self.void_cloud.points = o3d.utility.Vector3dVector([])
            self.void_cloud.colors = o3d.utility.Vector3dVector([])

        vis.update_geometry(self.void_cloud)
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
        print('  - Press "d" to change the visibility of the dynamic window')
        print('  - Press "v" to change the visibility of the void class')
        print('  - Press "s" to change the visibility of the static window')
        print('  - Press "c" to change the color of the static window')
        print('  - Press "q" to quit')
        self.update_window()
        self.update_scan()
        o3d.visualization.draw_geometries_with_key_callbacks(
            [self.static_cloud, self.scan, self.void_cloud, self.dynamic_cloud],
            self.key_callbacks)
