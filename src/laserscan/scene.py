try:
    from vispy.scene import visuals, ViewBox, SceneCanvas
except ImportError:
    visuals, ViewBox, SceneCanvas = None, None, None


class Scene:
    """ Scene for visualization.

    :param size: Size of the scene. (default: (800, 600))
    """

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
    """ Base class for widgets.

    :param scene: Scene object.
    :param pos: Position of the widget in the scene.
    :param color_map: Color map for the widget.
    :param border_color: Border color for the widget.
    """

    def __init__(self, scene: Scene, pos: tuple, color_map='viridis', border_color='white'):
        self.pos = pos
        self.scene = scene

        self.color_map = color_map
        self.view_box_keywords = dict(border_color=border_color)
        if scene is not None:
            self.view_box_keywords = dict(border_color=border_color, parent=scene.canvas.scene)


class CloudWidget(Widget):
    """ Widget for point cloud visualization.

    :param scene: Scene object.
    :param pos: Position of the widget in the scene.
    :param color_map: Color map for the widget.
    :param border_color: Border color for the widget.
    """

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
    """ Widget for image visualization.

    :param scene: Scene object.
    :param pos: Position of the widget in the scene.
    :param color_map: Color map for the widget.
    :param border_color: Border color for the widget.
    """

    def __init__(self, scene: Scene, pos: tuple, color_map='viridis', border_color='white'):
        super().__init__(scene, pos, color_map, border_color)
        self.vis = None
        self.view = None
        self.scene_grid = None

        if scene is not None:
            self._init()

    def _init(self):
        self.view = ViewBox(**self.view_box_keywords)
        self.scene.grid.add_widget(self.view, self.pos[0], self.pos[1])
        self.vis = visuals.Image()
        self.view.add(self.vis)

    def set_data(self, data):
        if self.vis is not None:
            self.vis.set_data(data)
            self.vis.update()


class Counter:
    """ Simple counter for iteration. Used in ScanVis for assigning
    widget positions.

    :param start: Start value of the counter. (default: -1)
    """

    def __init__(self, start=-1):
        self.value = start

    def __iter__(self):
        return self

    def __next__(self):
        self.value += 1
        return self.value
