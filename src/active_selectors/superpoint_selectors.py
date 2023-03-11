class BaseSuperpointSelector(object):
    def __init__(self):
        pass

    def select(self, *args, **kwargs):
        raise NotImplementedError
