import os
from copy import deepcopy

from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def set_paths(cfg: DictConfig, output_dir: str) -> DictConfig:
    cfg.path.output = os.path.dirname(output_dir)
    for i in cfg.path:
        cfg.path[i] = to_absolute_path(cfg.path[i])
    return cfg


def remote_paths(cfg: DictConfig, root_dir: str) -> DictConfig:
    paths = deepcopy(cfg.path)
    for name, path in cfg.path.items():
        paths[name] = path.replace(cfg.path.root, root_dir)
    return paths
