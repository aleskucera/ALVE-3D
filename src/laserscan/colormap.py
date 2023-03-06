import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt


def instances_color_map():
    # make instance colors
    max_inst_id = 100000
    color_map = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    # force zero to a gray-ish color
    color_map[0] = np.full(3, 0.1)
    return color_map


def map_color(data, color_map='viridis', data_range=(0, 19), bad=(0,)):
    cmap = plt.get_cmap(color_map)
    cmap.set_bad(color='black')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # # Normalize data
    data = (data - data_range[0]) / (data_range[1] - data_range[0])
    for b in bad:
        data = np.ma.masked_where(data == b, data)

    # Map data to color
    color = m.to_rgba(data)
    return color[..., :3]


def dict_to_color_map(dictionary: dict) -> np.ndarray:
    """
    Convert a dictionary to a color map
    :param dictionary: Dictionary with keys as indices and values as colors
    :return: Color map
    """
    size = max(dictionary.keys()) + 1
    color_map = np.zeros((size, 3), dtype=np.float32)
    for key, value in dictionary.items():
        color_map[key] = np.array(value, np.float32) / 255
    return color_map
