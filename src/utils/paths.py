import os
from hydra.utils import to_absolute_path


def paths_to_absolute(cfg):
    for i in cfg.path:
        cfg.path[i] = to_absolute_path(cfg.path[i])
    return cfg


def set_paths(cfg, output_dir):
    cfg.path.output = os.path.dirname(output_dir)
    for i in cfg.path:
        cfg.path[i] = to_absolute_path(cfg.path[i])
    return cfg
