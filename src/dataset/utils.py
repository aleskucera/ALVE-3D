import os
import h5py
import numpy as np


def open_sequence(path: str, split: str = None):
    velodyne_path = os.path.join(os.path.join(path, 'velodyne'))
    labels_path = os.path.join(os.path.join(path, 'labels'))

    poses_path = os.path.join(os.path.join(path, 'poses.txt'))
    calib_path = os.path.join(os.path.join(path, 'calib.txt'))

    info_path = os.path.join(path, 'info.h5')

    velodyne = []
    if os.path.exists(velodyne_path):
        velodyne = [os.path.join(velodyne_path, point) for point in os.listdir(velodyne_path)]
        velodyne.sort()

    labels = []
    if os.path.exists(labels_path):
        labels = [os.path.join(labels_path, label) for label in os.listdir(labels_path)]
        labels.sort()

    if os.path.exists(info_path):
        # info = np.load(info_path, allow_pickle=True)
        # poses = [pose for pose in info['poses']]
        # if split is not None:
        #     split_indices = info[split]
        #     velodyne = [velodyne[i] for i in split_indices]
        #     labels = [labels[i] for i in split_indices]
        #     poses = [poses[i] for i in split_indices]
        with h5py.File(info_path, 'r') as f:
            poses = list(f['poses'])
            if split is not None:
                split_indices = list(f[split])
                velodyne = [velodyne[i] for i in split_indices]
                labels = [labels[i] for i in split_indices]
                poses = [poses[i] for i in split_indices]
    else:
        calib = parse_calibration(calib_path)
        poses = parse_poses(poses_path, calib)

    return velodyne, labels, poses


def parse_calibration(path: str) -> dict:
    """ Parse calibration file
    :param path: path to the file
    :return: dictionary containing the calibration
    """
    calib = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                key, content = line.strip().split(":")
                data = [float(v) for v in content.strip().split()]
                calib[key] = create_transform_matrix(data)
        calib['Tr_cam_to_velo'] = np.array([[2.34773698e-04, -9.99944155e-01, -1.05634778e-02, 5.93721868e-02],
                                            [1.04494074e-02, 1.05653536e-02, -9.99889574e-01, -7.51087914e-02],
                                            [9.99945389e-01, 1.24365378e-04, 1.04513030e-02, -2.72132796e-01],
                                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return calib


def create_transform_matrix(data: list) -> np.ndarray:
    """ Create a transformation matrix from a data list
    :param data: a list containing the translation and rotation of length 12
    :return: transformation matrix
    """
    if len(data) == 12:
        matrix = np.eye(4)
        matrix[:3, :] = np.reshape(data, (3, 4))
    else:
        raise ValueError("Invalid data length")
    return matrix


def parse_poses(path: str, calib: dict) -> list:
    """ Parse poses from a file
    :param path: path to the file
    :param calib: calibration dictionary
    :return: list of poses
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            data = [float(v) for v in line.strip().split()]
            pose = create_transform_matrix(data)
            if 'Tr_cam_to_velo' in calib:
                poses.append(np.matmul(pose, calib['Tr_cam_to_velo']))
    return poses


def parse_times(path: str) -> list:
    """ Read times from a file
    :param path: path to the file
    :return: times as a numpy array
    """
    times = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                times.append(float(line))
    return times
