from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def set_paths(cfg: DictConfig, output_dir: str) -> DictConfig:
    cfg.path.output = output_dir
    for i in cfg.path:
        cfg.path[i] = to_absolute_path(cfg.path[i])
    cfg.ds.path = to_absolute_path(cfg.ds.path)
    return cfg
