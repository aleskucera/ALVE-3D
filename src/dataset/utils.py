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


def horizontal_shift(img, shift):
    if shift > 0:
        img_shifted = np.zeros_like(img)
        img_shifted[..., :shift] = img[..., -shift:]
        img_shifted[..., shift:] = img[..., :-shift]
    else:
        img_shifted = img
    return img_shifted


def apply_augmentations(data, label):
    # with probability 0.5 flip from L to R image and mask
    if np.random.random() <= 0.5:
        data = np.fliplr(data.transpose((1, 2, 0)))
        data = data.transpose((2, 0, 1))
        label = np.fliplr(label)

    # add rotation around vertical axis (Z)
    n_inputs, H, W = data.shape
    shift = np.random.choice(range(W))
    data = horizontal_shift(data, shift=shift)
    label = horizontal_shift(label, shift=shift)

    return data.copy(), label.copy()
