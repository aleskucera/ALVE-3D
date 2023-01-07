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
    sp_path = os.path.join(os.path.join(path, 'superpoints'))

    points = [os.path.join(points_path, point) for point in os.listdir(points_path)]
    labels = [os.path.join(labels_path, label) for label in os.listdir(labels_path)]
    superpoints = [os.path.join(sp_path, sp) for sp in os.listdir(sp_path)]

    points.sort()
    labels.sort()
    superpoints.sort()

    calib = parse_calibration(calib_path)
    poses = parse_poses(poses_path, calib['Tr_cam_to_velo'])
    times = parse_times(times_path)

    return points, labels, poses, times, superpoints


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
    matrix = np.eye(4)
    matrix[:3, :] = np.reshape(data, (3, 4))
    return matrix


def parse_poses(path: str, transformation: np.ndarray) -> list:
    """ Parse poses from a file
    :param path: path to the file
    :param transformation: transformation matrix
    :return: list of poses
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            data = [float(v) for v in line.strip().split()]
            pose = create_transform_matrix(data)
            poses.append(np.matmul(pose, transformation))
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
