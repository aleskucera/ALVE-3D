import os
import numpy as np

from omegaconf import DictConfig

from .classes import Sequence


def init_sequences(path: str, seq_list: list, seq_structure: DictConfig) -> list:
    """ Initialize the sequences
    :param path: path to the sequences
    :param seq_list: list of sequences to load
    :param seq_structure: structure of the sequences
    :return: list of Sequence objects
    """
    sequences = []
    for seq in seq_list:
        seq_name = f"{seq:02d}"
        seq_path = os.path.join(path, seq_name)
        sequences.append(Sequence(name=seq_name,
                                  path=seq_path,
                                  points_dir=os.path.join(seq_path, seq_structure.points_dir),
                                  labels_dir=os.path.join(seq_path, seq_structure.labels_dir),
                                  calib_file=os.path.join(seq_path, seq_structure.calib_file),
                                  poses_file=os.path.join(seq_path, seq_structure.poses_file),
                                  times_file=os.path.join(seq_path, seq_structure.times_file)))
    return sequences


def read_sequence_data(sequence: Sequence) -> tuple:
    """ Read a sequence from the directory
    :param sequence:
    :return: tuple of ids, points, labels, poses, times
    """
    ids = sorted([get_id(f) for f in os.listdir(sequence.points_dir)])
    points_files = read_files(sequence.points_dir, '.bin')
    labels_files = read_files(sequence.labels_dir, '.label')
    calib = parse_calibration(sequence.calib_file)
    poses = parse_poses(sequence.poses_file, calib['Tr_cam2velo'])
    times = parse_times(sequence.times_file)
    return ids, points_files, labels_files, calib, poses, times


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


def create_transform_matrix(data: list) -> np.ndarray:
    """ Create a transformation matrix from a data list
    :param data: list containing the translation and rotation of length 12
    :return: transformation matrix
    """
    matrix = np.eye(4)
    matrix[:3, :] = np.reshape(data, (3, 4))
    return matrix


def read_files(path: str, ext: str) -> list:
    """ Read all files from a directory
    :param path: path to the directory
    :param ext: extension of the files
    :return: list of files
    """
    files = sorted(os.listdir(path))
    return [os.path.join(path, f) for f in files if f.endswith(ext)]


def read_poses(path: str) -> list:
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


def get_id(path: str) -> int:
    """ Get the id of a point cloud from its path
    example: path/to/cloud/000000.bin -> 0
    :param path: path to the file
    :return: id of the file as int
    """
    return int(os.path.splitext(os.path.basename(path))[0])
