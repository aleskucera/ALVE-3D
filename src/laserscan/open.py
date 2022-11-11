import os
import numpy as np

SCAN_EXTENSION = '.bin'
LABEL_EXTENSION = '.label'
PREDICTION_EXTENSION = '.pred'
ENTROPY_EXTENSION = '.entropy'


# decorator for checking if the file exists
def check_file_exists(func):
    def wrapper(filename, *args):
        if not isinstance(filename, str):
            raise TypeError(f"Filename should be string type, but was {type(filename)}")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        return func(filename, *args)

    return wrapper


@check_file_exists
def open_scan(filename):
    if not filename.endswith(SCAN_EXTENSION):
        raise RuntimeError("Filename extension is not valid point cloud file.")
    cloud = np.fromfile(filename, dtype=np.float32)
    cloud = cloud.reshape((-1, 4))
    points = cloud[:, :3]
    remissions = cloud[:, 3]
    return points, remissions


@check_file_exists
def open_label(filename, size=None):
    """ Open label file and parse it into semantic and instance labels
    :param filename: path to label file
    :param size: size of the point cloud
    :return: semantic labels, instance labels
    """
    # check extension is a label
    if not filename.endswith(LABEL_EXTENSION):
        raise RuntimeError("Filename extension is not valid label file.")

    # open label
    label = np.fromfile(filename, dtype=np.int32)

    # parse label
    sem_label, inst_label = parse_label_data(label, size)
    return sem_label, inst_label


def parse_label_data(label, size=None):
    """ Parse label data into semantic and instance labels
    :param label: (numpy array) label data
    :param size: (int) number of points in the point cloud
    :return: (numpy array) semantic label, (numpy array) instance label
    """

    # check size
    if size is not None:
        if label.shape[0] != size:
            raise RuntimeError(f"Label size {label.shape[0]} does not match point cloud size {size}.")

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    return sem_label, inst_label


@check_file_exists
def open_prediction(filename, size=None):
    """ Open prediction file
    :param filename: (str) filename
    :param size: (int) number of points in the point cloud
    :return: (numpy array) prediction
    """
    # check extension is a prediction
    if not filename.endswith(PREDICTION_EXTENSION):
        raise RuntimeError("Filename extension is not valid prediction file.")

    # open prediction
    prediction = np.fromfile(filename, dtype=np.int32)

    # check size
    if size is not None:
        if prediction.shape[0] != size:
            raise RuntimeError(f"Prediction size {prediction.shape[0]} does not match point cloud size {size}.")
    return prediction


@check_file_exists
def open_entropy(filename, size=None):
    """ Open entropy file
    :param filename: (str) filename
    :param size: (int) number of points in the point cloud
    :return: (numpy array) entropy
    """
    # check extension is an entropy
    if not filename.endswith(ENTROPY_EXTENSION):
        raise RuntimeError("Filename extension is not valid entropy file.")

    # open entropy
    entropy = np.fromfile(filename, dtype=np.float32)

    # check size
    if size is not None:
        if entropy.shape[0] != size:
            raise RuntimeError(f"Entropy size {entropy.shape[0]} does not match point cloud size {size}.")
    return entropy
