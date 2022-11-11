import numpy as np

from .scan import LaserScan
from .open import open_label, parse_label_data
from .colormap import dict_to_color_map, instances_color_map


# wrapper that does projection if needed
def process_label(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if self.project:
            self.project_label()

        if self.colorize:
            self.colorize_label()

    return wrapper


class SemLaserScan(LaserScan):

    def __init__(self, project=True, colorize=False, sem_color_dict: dict = None,
                 H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        super(SemLaserScan, self).__init__(project, colorize, H, W, fov_up, fov_down)

        if colorize:
            assert sem_color_dict is not None, "Colorize arg is True but no color dict is given"
            self.sem_color_map = dict_to_color_map(sem_color_dict)
            self.inst_color_map = instances_color_map()

        # Cloud attributes
        self.sem_label = None
        self.sem_label_color = None

        self.inst_label = None
        self.inst_label_color = None

        # Projected attributes
        self.proj_sem_label = None
        self.proj_sem_color = None

        self.proj_inst_label = None
        self.proj_inst_color = None

    @process_label
    def open_label(self, filename):
        self.sem_label, self.inst_label = open_label(filename, self.points.shape[0])

    @process_label
    def set_label(self, label: np.ndarray):
        self.sem_label, self.inst_label = parse_label_data(label, self.points.shape[0])

    def colorize_label(self):
        self.sem_label_color = self.sem_color_map[self.sem_label]
        self.inst_label_color = self.inst_color_map[self.inst_label]

        self.proj_sem_color = self.sem_color_map[self.proj_sem_label]
        self.proj_inst_color = self.inst_color_map[self.proj_inst_label]

    def project_label(self):
        proj_indices = self.proj_idx[self.proj_mask]

        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

        self.proj_sem_label[self.proj_mask] = self.sem_label[proj_indices]
        self.proj_inst_label[self.proj_mask] = self.inst_label[proj_indices]
