#!/usr/bin/env python3
try:
    from vispy import app
except ImportError:
    app = None

import matplotlib.pyplot as plt
from .scan import LaserScan
from .scene import Scene, CloudWidget, ImageWidget, Counter


class ScanVis:
    def __init__(self, scan: LaserScan, scans: iter, scan_colors: iter = None, labels: iter = None,
                 superpoints: iter = None, predictions: iter = None, entropies: iter = None, offset: int = 0,
                 raw_cloud: bool = False, semantics: bool = True, instances: bool = False, projection: bool = True):

        self.scan = scan
        assert scan.colorize, "Scan must be colorized"

        self.scans = scans
        self.scan_colors = scan_colors
        self.labels = labels
        self.entropies = entropies
        self.predictions = predictions
        self.superpoints = superpoints
        self.total = len(scans)

        self.offset = offset

        self.raw_cloud = raw_cloud
        self.projection = projection

        if labels is not None:
            self.semantics = semantics
            self.instances = instances
        else:
            self.semantics = False
            self.instances = False

        # Point Cloud Scene
        self.scene = Scene()
        self.scene.connect(self.key_press, self.draw)

        # Image Scene
        self.img_scene = None
        if self.projection:
            self.img_scene = Scene(size=(scan.proj_W, scan.proj_H * self.num_widgets))
            self.img_scene.connect(self.key_press, self.draw)

        c = Counter()

        # Raw Cloud
        if self.raw_cloud:
            idx = next(c)
            self.scan_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        # Semantics
        if self.semantics:
            idx = next(c)
            self.sem_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.sem_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        # Instances
        if self.instances:
            idx = next(c)
            self.inst_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.inst_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        # Entropy
        if self.entropies is not None:
            idx = next(c)
            self.entropy_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.entropy_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        # Predictions
        if self.predictions is not None:
            idx = next(c)
            self.pred_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.pred_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        # Superpoints
        if self.superpoints is not None:
            idx = next(c)
            self.superpoint_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.superpoint_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        self.update_scan()

    def update_scan(self):
        # Update the title
        title = f"Scan {self.offset} / {self.total}"
        self.scene.update_title(title)
        if self.img_scene is not None:
            self.img_scene.update_title(title)

        # Update the scan
        color = self.scan_colors[self.offset] if self.scan_colors is not None else None
        if isinstance(self.scans[0], str):
            self.scan.open_scan(self.scans[self.offset], color=color)
        else:
            self.scan.set_scan(self.scans[self.offset], color=color)

        if self.raw_cloud:
            self.scan_w.set_data(self.scan.points, self.scan.color)
            self.img_w.set_data(self.scan.proj_color)

            # Save the pojection image
            plt.imsave(f"proj_{self.offset}.png", self.scan.proj_color)

        # Update the labels
        if self.labels is not None:
            if isinstance(self.labels[0], str):
                self.scan.open_label(self.labels[self.offset])
            else:
                self.scan.set_label(self.labels[self.offset])

            if self.semantics:
                self.sem_w.set_data(self.scan.points, self.scan.sem_label_color[..., ::-1])
                self.sem_img_w.set_data(self.scan.proj_sem_color[..., ::-1])
            if self.instances:
                self.inst_w.set_data(self.scan.points, self.scan.inst_label_color[..., ::-1])
                self.inst_img_w.set_data(self.scan.proj_inst_color[..., ::-1])

        # Update the entropies
        if self.entropies is not None:
            if isinstance(self.entropies[0], str):
                self.scan.open_entropy(self.entropies[self.offset])
            else:
                self.scan.set_entropy(self.entropies[self.offset])

            self.entropy_w.set_data(self.scan.points, self.scan.entropy_color[..., ::-1])
            self.entropy_img_w.set_data(self.scan.proj_entropy_color[..., ::-1])

        # Update the predictions
        if self.predictions is not None:
            if isinstance(self.predictions[0], str):
                self.scan.open_prediction(self.predictions[self.offset])
            else:
                self.scan.set_prediction(self.predictions[self.offset])

                self.pred_w.set_data(self.scan.points, self.scan.pred_color[..., ::-1])
                self.pred_img_w.set_data(self.scan.proj_pred_color[..., ::-1])

        # Update the superpoints
        if self.superpoints is not None:
            if isinstance(self.superpoints[0], str):
                self.scan.open_superpoints(self.superpoints[self.offset])
            else:
                self.scan.set_superpoints(self.superpoints[self.offset])

            self.superpoint_w.set_data(self.scan.points, self.scan.superpoints_color)
            self.superpoint_img_w.set_data(self.scan.proj_superpoints_color)

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
        if self.raw_cloud:
            num += 1
        if self.semantics:
            num += 1
        if self.instances:
            num += 1
        if self.entropies is not None:
            num += 1
        if self.predictions is not None:
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
