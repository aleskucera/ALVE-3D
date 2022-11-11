#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from vispy import app
from utils import Scene, CloudWidget, ImageWidget, map_color


class LaserScanVis:
    def __init__(self, scan, scan_names, label_names, offset: int = 0,
                 semantics: bool = True, instances: bool = False):

        self.scan = scan
        self.scan_names = scan_names
        self.label_names = label_names
        self.total = len(scan_names)

        self.offset = offset

        self.semantics = semantics
        self.instances = instances

        # sanity check
        if not self.semantics and self.instances:
            print("Instances are only allowed in when semantics=True")
            raise ValueError

        self.action = 'no'

        # Point Cloud Scene
        self.scene = Scene()
        self.scene.connect(self.key_press, self.draw)

        # Image Scene
        self.img_scene = Scene(size=(1024, 64 * self.multiplier()))
        self.img_scene.connect(self.key_press, self.draw)

        # Raw
        self.scan_w = CloudWidget(scene=self.scene, pos=(0, 0))
        self.img_w = ImageWidget(scene=self.img_scene, pos=(0, 0))

        # Semantics
        if self.semantics:
            self.sem_w = CloudWidget(scene=self.scene, pos=(0, 1))
            self.sem_img_w = ImageWidget(scene=self.img_scene, pos=(1, 0))

        # Instances
        if self.instances:
            self.inst_w = CloudWidget(scene=self.scene, pos=(0, 2))
            self.inst_img_w = ImageWidget(scene=self.img_scene, pos=(2, 0))

        self.update_scan()

    def update_scan(self):
        self.scan.open_scan(self.scan_names[self.offset])

        if self.semantics:
            self.scan.open_label(self.label_names[self.offset])
            self.scan.colorize()

        # Update the title
        title = f"Scan {self.offset}"
        self.scene.update_title(title)
        self.img_scene.update_title(title)

        # plot scan
        color = map_color(self.scan.unproj_range, vmin=-0.5, vmax=1.5, color_map='viridis')
        self.scan_w.set_data(self.scan.points, color)

        img_color = map_color(self.scan.proj_range, vmin=-0.2, vmax=1, color_map='jet')
        self.img_w.set_data(img_color)

        if self.semantics:
            self.sem_w.set_data(self.scan.points, self.scan.sem_label_color[..., ::-1])
            self.sem_img_w.set_data(self.scan.proj_sem_color[..., ::-1])
        if self.instances:
            self.inst_w.set_data(self.scan.points, self.scan.inst_label_color[..., ::-1])
            self.inst_img_w.set_data(self.scan.proj_inst_color[..., ::-1])

    def key_press(self, event):
        self.scene.canvas.events.key_press.block()
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
        if self.img_scene.canvas.events.key_press.blocked():
            self.img_scene.canvas.events.key_press.unblock()

    def multiplier(self):
        multiplier = 1
        if self.semantics:
            multiplier += 1
        if self.instances:
            multiplier += 1
        return multiplier

    def destroy(self):
        self.scene.canvas.close()
        self.img_scene.canvas.close()
        app.quit()

    @staticmethod
    def run():
        app.run()
