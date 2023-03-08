import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt


def map_labels(data: np.ndarray, mapping: dict):
    label_map = np.zeros(max(mapping.keys()) + 1, dtype=np.uint8)
    for label, value in mapping.items():
        label_map[label] = value
    return label_map[data]


def map_colors(data: np.ndarray, mapping: dict):
    color_map = np.zeros((max(mapping.keys()) + 1, 3), dtype=np.float32)
    for key, color in mapping.items():
        color_map[key] = np.array(color, np.float32) / 255
    return color_map[data]


def colorize_instances(data: np.ndarray, max_inst_id=100000, ignore: tuple = (0,)):
    """Colorize instances

    :param data: Data to colorize
    :param max_inst_id: Maximum instance id
    :param ignore: Ignore values (set to dark gray)
    """
    color_map = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    color_map[ignore] = np.full(3, 0.1)
    return color_map[data]


def colorize(data: np.ndarray, color_map: str = 'viridis', data_range: tuple = (0, 19),
             ignore: tuple = (0,)) -> np.ndarray:
    """Colorize data using a color map

    :param data: Data to colorize
    :param color_map: Color map to use
    :param data_range: Range of data
    :param ignore: Ignore values (set to dark gray)
    """
    # Create color map
    cmap = plt.get_cmap(color_map)
    cmap.set_bad(color=np.full(3, 0.1))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Normalize data
    data = (data - data_range[0]) / (data_range[1] - data_range[0])
    for i in ignore:
        data = np.ma.masked_where(data == i, data)

    # Map data to color
    color = m.to_rgba(data)
    return color[..., :3]
