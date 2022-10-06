import numpy as np


def apply_augmentations(data: np.ndarray, label: np.ndarray) -> tuple:
    """ Apply augmentations to the data (flip and shift)
    :param data: data
    :param label: labels
    :return:
    """
    flip_sample(data, label, prob=0.5)
    shift_sample(data, label)
    return data.copy(), label.copy()


def flip_sample(data: np.ndarray, label: np.ndarray, prob: float = 0.5) -> tuple:
    """ Flip the data
    :param data: data
    :param label: labels
    :param prob: probability of flipping
    :return: flipped data and labels
    """
    data = np.fliplr(data.transpose((1, 2, 0)))
    data = data.transpose((2, 0, 1))
    label = np.fliplr(label)
    return data, label


def shift_sample(data: np.ndarray, label: np.ndarray) -> tuple:
    """ Shift the data
    :param data: data
    :param label: labels
    :return: shifted data and labels
    """
    n_inputs, H, W = data.shape
    shift = np.random.choice(range(W))
    data = horizontal_shift(data, shift=shift)
    label = horizontal_shift(label, shift=shift)
    return data, label


def horizontal_shift(img: np.ndarray, shift: int) -> np.ndarray:
    """ Shift the image horizontally
    :param img: image
    :param shift: shift
    :return: shifted image
    """
    if shift > 0:
        img_shifted = np.zeros_like(img)
        img_shifted[..., :shift] = img[..., -shift:]
        img_shifted[..., shift:] = img[..., :-shift]
    else:
        img_shifted = img
    return img_shifted
