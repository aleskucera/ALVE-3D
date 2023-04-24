#!/usr/bin/env python3
try:
    from vispy import app
except ImportError:
    app = None

import h5py
import numpy as np
from .scan import LaserScan
from .scene import Scene, CloudWidget, ImageWidget, Counter


class ScanVis:
    """Visualize scans and labels in 3D and 2D.

    :param laser_scan: LaserScan object.
    :param scans: List of scans to visualize.
    :param labels: List of labels to visualize. If None, no labels will be visualized. (Default: None)
    :param predictions: List of predictions to visualize. If None, no predictions will be visualized. (Default: None)
    :param projection: Whether to project the scans and labels. (Default: True)
    :param raw_scan: Whether to visualize the raw scan. (Default: True)
    :param offset: Offset of the scan in the dataset. (Default: 0)
    """

    def __init__(self, laser_scan: LaserScan, scans: iter, labels: iter = None, predictions: iter = None,
                 projection: bool = True, raw_scan: bool = True, offset: int = 0):

        assert laser_scan.colorize, "Scan must be colorized"
        self.laser_scan = laser_scan

        self.scans = scans
        self.labels = labels
        self.pred = predictions

        self.offset = offset
        self.total = len(scans)

        self.raw_scan = raw_scan
        self.projection = projection

        # ================= VISUALIZATION SCENES =================

        # Point Cloud Scene
        self.scene = Scene()
        self.scene.connect(self.key_press, self.draw)

        # Image Scene
        self.img_scene = None
        if self.projection:
            self.img_scene = Scene(size=(laser_scan.proj_W, laser_scan.proj_H * self.num_widgets))
            self.img_scene.connect(self.key_press, self.draw)

        # ================= VISUALIZATION WIDGETS (WINDOWS) =================

        c = Counter()

        # Raw Cloud
        if self.raw_scan:
            idx = next(c)
            self.scan_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.scan_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        # Labels
        if self.labels is not None:
            idx = next(c)
            self.label_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.label_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        # Predictions
        if self.pred is not None:
            idx = next(c)
            self.pred_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.pred_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        self.update_scan()

    def update_scan(self):

        # ======================== TITLE ========================

        title = f"Scan {self.offset} / {self.total}"
        self.scene.update_title(title)
        if self.img_scene is not None:
            self.img_scene.update_title(title)

        # ======================== SCAN ========================

        if isinstance(self.scans[0], str):
            self.laser_scan.open_scan(self.scans[self.offset])
        else:
            self.laser_scan.set_scan(self.scans[self.offset])

        if self.raw_scan:
            self.scan_w.set_data(self.laser_scan.points, self.laser_scan.color)
            self.scan_img_w.set_data(self.laser_scan.proj_color)

        # ======================== LABELS ========================

        if self.labels is not None:
            if isinstance(self.labels[0], str):
                self.laser_scan.open_label(self.labels[self.offset])
            else:
                self.laser_scan.set_label(self.labels[self.offset])

            self.label_w.set_data(self.laser_scan.points, self.laser_scan.label_color)
            self.label_img_w.set_data(self.laser_scan.proj_label_color)

        # ======================== PREDICTIONS ========================
        if self.pred is not None:
            if isinstance(self.pred[0], str):
                self.laser_scan.open_prediction(self.pred[self.offset])
            else:
                self.laser_scan.set_prediction(self.pred[self.offset])

            self.pred_w.set_data(self.laser_scan.points, self.laser_scan.pred_color)
            self.pred_img_w.set_data(self.laser_scan.proj_pred_color)

    def key_press(self, event):
        self.scene.canvas.events.key_press.block()
        if self.img_scene is not None:
            self.img_scene.canvas.events.key_press.block()

        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0
            self.update_scan()
        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1
            self.update_scan()
        elif event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def draw(self, event):
        if self.scene.canvas.events.key_press.blocked():
            self.scene.canvas.events.key_press.unblock()
        if self.img_scene is not None and self.img_scene.canvas.events.key_press.blocked():
            self.img_scene.canvas.events.key_press.unblock()

    @property
    def num_widgets(self):
        num = 0
        if self.raw_scan:
            num += 1
        if self.labels is not None:
            num += 1
        return num

    def destroy(self):
        self.scene.canvas.close()
        if self.img_scene is not None:
            self.img_scene.canvas.close()
        app.quit()

    @staticmethod
    def run():
        print("\nControls:")
        print("\tN: Next Scan")
        print("\tB: Previous Scan")
        print("\tQ: Quit\n")
        app.run()
