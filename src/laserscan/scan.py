#!/usr/bin/env python3
import numpy as np
from .open import open_scan
from .project import project_scan
from .colormap import map_color


def process_data(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if self.project:
            self.project_data()

        if self.colorize:
            self.colorize_data()

    return wrapper


class LaserScan:
    def __init__(self, project=True, colorize=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        self.proj_H = H
        self.proj_W = W

        self.project = project
        self.colorize = colorize

        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

        self.points = None
        self.color = None
        self.remissions = None
        self.radius = None

        self.proj_color = None
        self.proj_depth = None
        self.proj_xyz = None
        self.proj_remission = None
        self.proj_idx = None
        self.proj_mask = None

    def __len__(self):
        return self.points.shape[0]

    @process_data
    def open_scan(self, filename) -> None:
        self.points, self.remissions = open_scan(filename)
        self.radius = np.linalg.norm(self.points, axis=1)

    @process_data
    def set_points(self, points, remissions=None):
        if remissions is None:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        self.points = points
        self.remissions = remissions
        self.radius = np.linalg.norm(self.points, axis=1)

    def colorize_data(self):
        self.color = map_color(self.radius, vmin=-0.5, vmax=1.5, color_map='viridis')
        self.proj_color = map_color(self.proj_depth, vmin=-0.2, vmax=1, color_map='jet')

    def project_data(self):
        projection = project_scan(self.points, self.remissions, self.proj_H,
                                  self.proj_W, self.proj_fov_up, self.proj_fov_down)

        self.proj_depth = projection['depth']
        self.proj_xyz = projection['xyz']
        self.proj_remission = projection['remission']
        self.proj_idx = projection['idx']
        self.proj_mask = projection['mask']
