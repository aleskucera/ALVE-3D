from vispy.scene import visuals, SceneCanvas, ViewBox


class Scene:
    def __init__(self, size=(800, 600)):
        self.canvas = SceneCanvas(keys='interactive', show=True, size=size)
        self.grid = self.canvas.central_widget.add_grid()

    def connect(self, key_press_callback, draw_callback):
        assert self.canvas is not None, "Canvas must be set"
        self.canvas.events.key_press.connect(key_press_callback)
        self.canvas.events.draw.connect(draw_callback)

    def update_title(self, title):
        self.canvas.title = title


class Widget:
    def __init__(self, scene: Scene, pos: tuple, color_map='viridis', border_color='white'):
        self.scene = scene
        self.pos = pos

        self.color_map = color_map
        self.view_box_keywords = {'border_color': border_color, 'parent': scene.canvas.scene}


class CloudWidget(Widget):
    def __init__(self, scene: Scene, pos: tuple, color_map='viridis', border_color='white'):
        super().__init__(scene, pos, color_map, border_color)
        self._init()

    def _init(self):
        self.view = ViewBox(**self.view_box_keywords)
        self.scene.grid.add_widget(self.view, self.pos[0], self.pos[1])
        self.vis = visuals.Markers()
        self.view.camera = 'turntable'
        self.view.add(self.vis)
        visuals.XYZAxis(parent=self.view.scene)

    def set_data(self, scan, color):
        self.vis.set_data(scan, face_color=color, edge_color=color, size=1)
        self.vis.update()


class ImageWidget(Widget):
    def __init__(self, scene: Scene, pos: tuple, color_map='viridis', border_color='white'):
        super().__init__(scene, pos, color_map, border_color)
        self._init()

    def _init(self):
        self.view = ViewBox(**self.view_box_keywords)
        self.scene.grid.add_widget(self.view, self.pos[0], self.pos[1])
        self.vis = visuals.Image()
        self.view.add(self.vis)

    def set_data(self, data):
        self.vis.set_data(data)
        self.vis.update()


class Counter:
    def __init__(self, start=0):
        self.value = start

    def __iter__(self):
        return self

    def __next__(self):
        self.value += 1
        return self.value
