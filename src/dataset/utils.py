import os
import numpy as np


def dict_to_label_map(map_dict: dict) -> np.ndarray:
    label_map = np.zeros(max(map_dict.keys()) + 1, dtype=np.uint8)
    for label, value in map_dict.items():
        label_map[label] = value
    return label_map


def open_sequence(path: str):
    points_path = os.path.join(os.path.join(path, 'velodyne'))
    labels_path = os.path.join(os.path.join(path, 'labels'))
    poses_path = os.path.join(os.path.join(path, 'poses.txt'))
    times_path = os.path.join(os.path.join(path, 'times.txt'))
    calib_path = os.path.join(os.path.join(path, 'calib.txt'))

    points = os.listdir(points_path)
    labels = os.listdir(labels_path)

    points = [os.path.join(points_path, point) for point in points]
    labels = [os.path.join(labels_path, label) for label in labels]

    poses = parse_poses(poses_path)
    times = parse_times(times_path)
    calib = parse_calibration(calib_path)

    points.sort()
    labels.sort()

    return points, labels, poses, times, calib


def parse_calibration(path: str) -> dict:
    """ Parse calibration file
    :param path: path to the file
    :return: dictionary containing the calibration
    """
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            key, content = line.strip().split(":")
            data = [float(v) for v in content.strip().split()]
            calib[key] = create_transform_matrix(data)
    calib['Tr_cam2velo'] = np.array([[2.34773698e-04, -9.99944155e-01, -1.05634778e-02, 5.93721868e-02],
                                     [1.04494074e-02, 1.05653536e-02, -9.99889574e-01, -7.51087914e-02],
                                     [9.99945389e-01, 1.24365378e-04, 1.04513030e-02, -2.72132796e-01],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return calib


def create_transform_matrix(data: list) -> np.ndarray:
    """ Create a transformation matrix from a data list
    :param data: list containing the translation and rotation of length 12
    :return: transformation matrix
    """
    matrix = np.eye(4)
    matrix[:3, :] = np.reshape(data, (3, 4))
    return matrix


def parse_poses(path: str) -> list:
    """ Read poses from a file
    :param path: path to the file
    :return: poses as a numpy array
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            poses.append([float(x) for x in line.split()])
    return poses


def parse_times(path: str) -> list:
    """ Read times from a file
    :param path: path to the file
    :return: times as a numpy array
    """
    times = []
    with open(path, 'r') as f:
        for line in f:
            times.append(float(line))
    return times
