import numpy as np

from .scan import LaserScan
from .open import open_entropy
from .colormap import map_color


def process_entropy(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if self.project:
            self.project_entropy()

        if self.colorize:
            self.colorize_entropy()

    return wrapper


class EntropyLaserScan(LaserScan):

    def __init__(self, project=True, colorize=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        super(EntropyLaserScan, self).__init__(project, colorize, H, W, fov_up, fov_down)
        self.entropy_color_map = {}

        self.entropy = None
        self.entropy_color = None

        self.proj_entropy = None
        self.proj_entropy_color = None

    @process_entropy
    def open_entropy(self, filename):
        self.entropy = open_entropy(filename, self.points.shape[0])

    @process_entropy
    def set_entropy(self, entropy: np.ndarray):
        self.entropy = entropy

    def colorize_entropy(self):
        self.entropy_color = map_color(self.entropy, vmin=-0.5, vmax=1.5, color_map='viridis')
        self.proj_entropy_color = map_color(self.proj_entropy, vmin=-0.5, vmax=1.5, color_map='viridis')

    def project_entropy(self):
        proj_indices = self.proj_idx[self.proj_mask]

        self.proj_entropy = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_entropy[self.proj_mask] = self.entropy[proj_indices]
