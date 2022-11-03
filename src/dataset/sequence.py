import os
import numpy as np

from .sample import Sample


class Sequence:
    def __init__(self, name: str, path: str, points_dir: str, labels_dir: str,
                 poses_file: str, times_file: str, calib_file: str):
        self.name = name
        self.path = path
        self.points_dir = points_dir
        self.labels_dir = labels_dir
        self.poses_file = poses_file
        self.times_file = times_file
        self.calib_file = calib_file

    def get_samples(self) -> list:
        ids = sorted([_get_id(f) for f in os.listdir(self.points_dir)])
        points_files = _read_file_names(self.points_dir, '.bin')
        labels_files = _read_file_names(self.labels_dir, '.label')
        calib = _parse_calibration(self.calib_file)
        poses = _parse_poses(self.poses_file, calib['Tr_cam2velo'])
        times = _parse_times(self.times_file)

        samples = []
        for i in ids:
            sample = Sample(id=i, time=times[i], points_path=points_files[i],
                            label_path=labels_files[i], pose=poses[i], calibration=calib)
            samples.append(sample)

        return samples

    def __repr__(self):
        return f"\nSEQUENCE NAME: {self.name}" \
               f"\n\tPath: {self.path}" \
               f"\n\tPoints_dir: {self.points_dir}" \
               f"\n\tLabels_dir: {self.labels_dir}" \
               f"\n\tPoses_file: {self.poses_file}" \
               f"\n\tTimes_file: {self.times_file}" \
               f"\n\tCalib_file: {self.calib_file}"


def _get_id(path: str) -> int:
    """ Get the id of a point cloud from its path
    example: path/to/cloud/000000.bin -> 0
    :param path: path to the file
    :return: id of the file as int
    """
    return int(os.path.splitext(os.path.basename(path))[0])


def _read_file_names(path: str, ext: str) -> list:
    """ Read all file names from a directory with a specific extension
    :param path: path to the directory
    :param ext: extension of the files
    :return: list of file names
    """
    files = sorted(os.listdir(path))
    return [os.path.join(path, f) for f in files if f.endswith(ext)]


def _parse_calibration(path: str) -> dict:
    """ Parse calibration file
    :param path: path to the file
    :return: dictionary containing the calibration
    """
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            key, content = line.strip().split(":")
            data = [float(v) for v in content.strip().split()]
            calib[key] = _create_transform_matrix(data)
    calib['Tr_cam2velo'] = np.array([[2.34773698e-04, -9.99944155e-01, -1.05634778e-02, 5.93721868e-02],
                                     [1.04494074e-02, 1.05653536e-02, -9.99889574e-01, -7.51087914e-02],
                                     [9.99945389e-01, 1.24365378e-04, 1.04513030e-02, -2.72132796e-01],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return calib


def _parse_poses(path: str, transformation: np.ndarray) -> list:
    """ Parse poses from a file
    :param path: path to the file
    :param transformation: transformation matrix
    :return: list of poses
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            data = [float(v) for v in line.strip().split()]
            pose = _create_transform_matrix(data)
            poses.append(np.matmul(pose, transformation))
    return poses


def _parse_times(path: str) -> list:
    """ Read times from a file
    :param path: path to the file
    :return: times as a numpy array
    """
    times = []
    with open(path, 'r') as f:
        for line in f:
            times.append(float(line))
    return times


def _create_transform_matrix(data: list) -> np.ndarray:
    """ Create a transformation matrix from a data list
    :param data: list containing the translation and rotation of length 12
    :return: transformation matrix
    """
    matrix = np.eye(4)
    matrix[:3, :] = np.reshape(data, (3, 4))
    return matrix
