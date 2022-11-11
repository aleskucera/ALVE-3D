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

    points = os.listdir(points_path)
    labels = os.listdir(labels_path)

    points = [os.path.join(points_path, point) for point in points]
    labels = [os.path.join(labels_path, label) for label in labels]

    points.sort()
    labels.sort()

    return points, labels
