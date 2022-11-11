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


def map_color(data, vmin=0, vmax=1, color_map='viridis'):
    cmap = plt.get_cmap(color_map)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Normalize data
    data = (data - data.min()) / (data.max() - data.min())

    # Map data to color
    color = m.to_rgba(data)
    return color


# def dict_colorize(data: np.ndarray, dictionary: dict):
#     """
#     Colorize numpy array based on a dictionary
#     :param data: Data to be colorized
#     :param dictionary: Dictionary with keys as indices and values as colors
#     :return: Colorized data
#     """
#     color_map = dict_to_color_map(dictionary)
#     return color_map[data]


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


if __name__ == '__main__':
    dict_map = {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255], 6: [255, 255, 0]}
    print(max(dict_map, key=dict_map.get))
    print(max(dict_map.keys()))

    arr = np.array([0, 1, 4, 6, 6])
    print("Old array: ", arr)
    cmp = dict_to_color_map(dict_map)
    print("Mapped array: ", cmp[arr])
