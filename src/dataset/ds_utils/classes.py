import numpy as np
from dataclasses import dataclass


@dataclass
class Sample:
    """Class representing a sample from the dataset"""
    id: int
    time: float
    label_path: str
    points_path: str
    pose: np.ndarray
    calibration: np.ndarray
    label: np.ndarray = None
    points: np.ndarray = None

    def __repr__(self):
        ret = f"\nSAMPLE ID: {self.id}" \
              f"\n\tTime: {self.time}" \
              f"\n\tLabel_path: {self.label_path}" \
              f"\n\tPoints_path: {self.points_path}" \
              f"\n\tPose: \n{self.pose}"
        if self.label is not None:
            ret += f"\n\tLabel: {self.label.shape}"
        if self.points is not None:
            ret += f"\n\tPoints: {self.points.shape}"
        return ret


@dataclass
class Sequence:
    """Class for storing sequence information"""
    name: str
    path: str
    points_dir: str
    labels_dir: str
    poses_file: str
    times_file: str
    calib_file: str
