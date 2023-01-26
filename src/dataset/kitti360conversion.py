import os

import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from omegaconf import DictConfig
from src.laserscan.project import project_scan
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.lib.recfunctions import structured_to_unstructured

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: Can't import open3d.")

try:
    from kitti360scripts.helpers.ply import read_ply
    from kitti360scripts.helpers.labels import id2label
    from kitti360scripts.helpers.annotation import Annotation3DPly, global2local

except ImportError:
    print("WARNING: Can't import kitti360scripts. Import kitti360scripts by running "
          "'pip install git+https://github.com/autonomousvision/kitti360Scripts.git'.")
    read_ply, id2label, Annotation3DPly, global2local = None, None, None, None


def visualize_kitti360_conversion(cfg: DictConfig, sequence: int):
    converter = Kitti360Converter(cfg, sequence)
    converter.run()


def convert_kitti360(cfg: DictConfig, sequence: int):
    converter = Kitti360Converter(cfg, sequence)
    converter.convert()


class Kitti360Converter:
    def __init__(self, cfg: DictConfig, sequence: int):
        # Sequence paths
        self.cfg = cfg
        self.sequence = sequence
        self.seq_name = f'2013_05_28_drive_{sequence:04d}_sync'

        self.velodyne_path = os.path.join(cfg.ds.path, 'data_3d_raw', self.seq_name, 'velodyne_points', 'data')
        self.semantics_path = os.path.join(cfg.ds.path, 'data_3d_semantics')

        self.poses_path = os.path.join(cfg.ds.path, 'data_poses', self.seq_name, 'cam0_to_world.txt')
        self.calib_path = os.path.join(cfg.ds.path, 'calibration', 'calib_cam_to_velo.txt')

        # Sequence windows
        static_windows_path = os.path.join(self.semantics_path, 'train', self.seq_name, 'static')
        dynamic_windows_path = os.path.join(self.semantics_path, 'train', self.seq_name, 'dynamic')

        self.static_windows = [os.path.join(static_windows_path, f) for f in os.listdir(static_windows_path)]
        self.dynamic_windows = [os.path.join(dynamic_windows_path, f) for f in os.listdir(dynamic_windows_path)]

        self.static_windows.sort()
        self.dynamic_windows.sort()

        self.window_ranges = [get_window_range(window) for window in self.static_windows]

        # Sequence info
        self.num_scans = len(os.listdir(self.velodyne_path))
        self.num_windows = len(self.static_windows)

        # Transformations
        self.T_cam2velo = np.concatenate([np.loadtxt(self.calib_path).reshape(3, 4), [[0, 0, 0, 1]]], axis=0)
        self.T_velo2cam = np.linalg.inv(self.T_cam2velo)
        self.poses = read_poses(self.poses_path, self.T_velo2cam, self.num_scans)

        # Point clouds for visualization
        self.scan = o3d.geometry.PointCloud()
        self.static_window = o3d.geometry.PointCloud()
        self.dynamic_window = o3d.geometry.PointCloud()

        self.static_points = None

        self.semantic = None
        self.instances = None

        self.semantic_color = None
        self.instance_color = None

        # Visualization parameters
        self.scan_num = 0
        self.window_num = 0

        self.cmap = cm.get_cmap('Set1')
        self.cmap_length = 9

        self.key_callbacks = {
            ord(']'): self.prev_window,
            ord(']'): self.next_window,
            ord('N'): self.next_scan,
            ord('B'): self.prev_scan,
            ord('Q'): self.quit,
        }

    def convert(self):
        # Create output directories
        sequence_path = os.path.join(self.cfg.ds.path, 'sequences', f'{self.sequence:02d}')

        velodyne_dir = os.path.join(sequence_path, 'velodyne')
        os.makedirs(velodyne_dir, exist_ok=True)

        labels_dir = os.path.join(sequence_path, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        poses_file = os.path.join(sequence_path, 'poses.txt')

        for i, window_file in enumerate(self.static_windows):
            print(f'Converting window {i + 1}/{self.num_windows}')
            window = read_ply(window_file)
            window_points = structured_to_unstructured(window[['x', 'y', 'z']])
            window_colors = structured_to_unstructured(window[['red', 'green', 'blue']]) / 255
            self.semantic = structured_to_unstructured(window[['semantic']])
            self.instances = structured_to_unstructured(window[['instance']])

            print(f'Window range: {self.window_ranges[i]}')
            start, end = self.window_ranges[i]
            for j in tqdm(range(start, end)):
                scan = read_scan(self.velodyne_path, j)
                scan_points = scan[:, :3]
                scan_intensity = scan[:, 3][:, np.newaxis]

                # Transform scan to world coordinates
                hom_scan_points = np.concatenate([scan_points, np.ones((scan_points.shape[0], 1))], axis=1)
                hom_scan_points = np.matmul(self.poses[j], hom_scan_points.T).T
                transformed_scan_points = hom_scan_points[:, :3]

                # Find neighbours in the static window
                tree = scipy.spatial.cKDTree(window_points)
                dists, indices = tree.query(transformed_scan_points, k=1)
                mask = np.logical_and(dists >= 0, dists <= 0.3)

                rgb = window_colors[indices[mask]]

                # Save the scan
                scan = np.concatenate([scan_points[mask], scan_intensity[mask], rgb], axis=1, dtype=np.float32)
                np.save(os.path.join(velodyne_dir, f'{j:06d}.npy'), scan)

                labels = np.zeros((scan.shape[0], 1), dtype=np.int32)
                semantics = self.semantic[indices[mask]].astype(np.int32)
                instances = self.instances[indices[mask]].astype(np.int32)

                labels = labels | semantics
                labels = labels | (instances << 16)

                np.save(os.path.join(labels_dir, f'{j:06d}.npy'), labels)

        # Save the poses
        np.save(poses_file, self.poses)

    def update_window(self):

        # Read static window
        static_window = read_ply(self.static_windows[self.window_num])

        static_points = structured_to_unstructured(static_window[['x', 'y', 'z']])
        static_colors = structured_to_unstructured(static_window[['red', 'green', 'blue']]) / 255

        self.semantic = structured_to_unstructured(static_window[['semantic']]).flatten()
        self.instances = structured_to_unstructured(static_window[['instance']]).flatten()

        # Read dynamic window
        dynamic_window = read_ply(self.dynamic_windows[self.window_num])
        dynamic_points = structured_to_unstructured(dynamic_window[['x', 'y', 'z']])
        dynamic_colors = np.ones_like(dynamic_points) * [0, 0, 1]

        # Colorize for visualization
        self.semantic_color = np.zeros((self.instances.size, 3))
        self.instance_color = np.zeros((self.instances.size, 3))

        for uid in np.unique(self.instances):
            semantic_id, instance_id = global2local(uid)
            self.semantic_color[self.instances == uid] = id2label[semantic_id].color
            if instance_id == 0:
                self.instance_color[self.instances == uid] = (0, 0, 0)
            elif instance_id > 0:
                self.instance_color[self.instances == uid] = np.asarray(self.cmap(instance_id % self.cmap_length)[:3])
            else:
                self.instance_color[self.instances == uid] = np.ndarray([96, 96, 96]) / 255.

        self.semantic_color = self.semantic_color / 255.

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
        hom_scan_points = np.concatenate([scan_points, np.ones((scan_points.shape[0], 1))], axis=1)
        hom_scan_points = np.matmul(self.poses[self.scan_num], hom_scan_points.T).T
        transformed_scan_points = hom_scan_points[:, :3]

        # Find neighbours in the static window
        tree = scipy.spatial.cKDTree(np.array(self.static_window.points))
        dists, indices = tree.query(transformed_scan_points, k=1)
        mask = np.logical_and(dists >= 0, dists <= 0.3)

        # Extract RGB values from the static window
        rgb = np.array(self.static_window.colors)[indices[mask]]

        # Color of the scan in world coordinates
        scan_colors = np.ones_like(scan_points) * [1, 0, 0]
        scan_colors[mask] = [0, 1, 0]

        # Get point cloud labels
        semantics = np.array(self.semantic_color)[indices[mask]]
        instances = np.array(self.instance_color)[indices[mask]]

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
        proj_instances[filtered_proj_mask] = instances[filtered_proj_indices]

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
        self.window_num -= 1
        self.update_window()
        vis.update_geometry(self.static_window)
        vis.update_geometry(self.dynamic_window)
        vis.update_geometry(self.scan)
        vis.reset_view_point(True)
        vis.update_renderer()
        return False

    def next_scan(self, vis):
        self.scan_num += 20
        self.update_scan()
        vis.update_geometry(self.scan)
        vis.update_renderer()
        return False

    def prev_scan(self, vis):
        self.scan_num -= 20
        self.update_scan()
        vis.update_geometry(self.scan)
        vis.update_renderer()
        return False

    @staticmethod
    def quit(vis):
        vis.destroy_window()
        return True

    def run(self):
        self.update_window()
        self.update_scan()
        o3d.visualization.draw_geometries_with_key_callbacks(
            [self.static_window, self.dynamic_window, self.scan],
            self.key_callbacks)


def read_poses(poses_path: str, T_velo2cam: np.ndarray, sequence_length: int) -> np.ndarray:
    """Read poses from poses.txt file. The poses are transformations from the velodyne coordinate
    system to the world coordinate system.
    :return: array of poses (Nx4x4), where N is the number of velodyne scans
    """

    # Load poses. Some poses are missing, because the camera was not moving.
    compressed_poses = np.loadtxt(poses_path, dtype=np.float32)
    frames = compressed_poses[:, 0].astype(np.int32)
    lidar_poses = compressed_poses[:, 1:].reshape(-1, 4, 4)

    # Create a full list of poses (with missing poses with value of the last known pose)
    poses = np.zeros((sequence_length, 4, 4), dtype=np.float32)

    last_valid_pose = lidar_poses[0]
    for i in range(sequence_length):
        if i in frames:
            last_valid_pose = lidar_poses[frames == i] @ T_velo2cam
        poses[i] = last_valid_pose
    return poses


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
