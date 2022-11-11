#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from vispy import app
from .scene import Scene, CloudWidget, ImageWidget


class Counter:
    def __init__(self, start=0):
        self.value = start

    def __iter__(self):
        return self

    def __next__(self):
        self.value += 1
        return self.value


class ScanVis:
    def __init__(self, scan, scan_names, label_names: list = None, offset: int = 0,
                 semantics: bool = False, instances: bool = False, entropy: bool = False):

        self.scan = scan
        assert scan.colorize, "Scan must be colorized"

        self.scan_names = scan_names
        self.label_names = label_names
        self.total = len(scan_names)

        self.offset = offset

        self.entropy = entropy
        self.semantics = semantics
        self.instances = instances

        self.action = 'none'

        # Point Cloud Scene
        self.scene = Scene()
        self.scene.connect(self.key_press, self.draw)

        # Image Scene
        self.img_scene = Scene(size=(1024, 64 * self.num_widgets))
        self.img_scene.connect(self.key_press, self.draw)

        c = Counter()

        # Raw
        idx = c.value
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

        if self.entropy:
            idx = next(c)
            self.entropy_w = CloudWidget(scene=self.scene, pos=(0, idx))
            self.entropy_img_w = ImageWidget(scene=self.img_scene, pos=(idx, 0))

        self.update_scan()

    def update_scan(self):
        self.scan.open_scan(self.scan_names[self.offset])

        if self.semantics or self.instances:
            self.scan.open_label(self.label_names[self.offset])

        # Update the title
        title = f"Scan {self.offset} / {self.total}"
        self.scene.update_title(title)
        self.img_scene.update_title(title)

        # plot scan
        self.scan_w.set_data(self.scan.points, self.scan.color)
        self.img_w.set_data(self.scan.proj_color)

        if self.semantics:
            self.sem_w.set_data(self.scan.points, self.scan.sem_label_color[..., ::-1])
            self.sem_img_w.set_data(self.scan.proj_sem_color[..., ::-1])
        if self.instances:
            self.inst_w.set_data(self.scan.points, self.scan.inst_label_color[..., ::-1])
            self.inst_img_w.set_data(self.scan.proj_inst_color[..., ::-1])
        if self.entropy:
            self.entropy_w.set_data(self.scan.points, self.scan.entropy_color[..., ::-1])
            self.entropy_img_w.set_data(self.scan.proj_entropy_color[..., ::-1])

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

    @property
    def num_widgets(self):
        return 1 + self.semantics + self.instances + self.entropy

    def destroy(self):
        self.scene.canvas.close()
        self.img_scene.canvas.close()
        app.quit()

    @staticmethod
    def run():
        app.run()
