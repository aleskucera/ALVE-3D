import numpy as np

from .scan import LaserScan
from .open import open_prediction
from .colormap import map_color


def process_prediction(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if self.project:
            self.project_prediction()

        if self.colorize:
            self.colorize_prediction()

    return wrapper


class PredictionLaserScan(LaserScan):

    def __init__(self, project=True, colorize=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        super(PredictionLaserScan, self).__init__(project, colorize, H, W, fov_up, fov_down)

        self.prediction = None
        self.prediction_color = None

        self.proj_prediction = None
        self.proj_prediction_color = None

    @process_prediction
    def open_prediction(self, filename):
        self.prediction = open_prediction(filename, self.points.shape[0])

    @process_prediction
    def set_prediction(self, prediction: np.ndarray):
        self.prediction = prediction

    def colorize_prediction(self):
        self.prediction_color = map_color(self.prediction, vmin=-0.5, vmax=1.5, color_map='viridis')
        self.proj_prediction_color = map_color(self.proj_prediction, vmin=-0.5, vmax=1.5, color_map='viridis')

    def project_prediction(self):
        proj_indices = self.proj_idx[self.proj_mask]

        self.proj_prediction = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        self.proj_prediction[self.proj_mask] = self.prediction[proj_indices]
