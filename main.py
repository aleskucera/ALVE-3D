import logging
import hydra
from omegaconf import DictConfig

from src import train_model, test_model, paths_to_absolute

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = paths_to_absolute(cfg)
    if cfg.action == 'train':
        train_model(cfg)
    elif cfg.action == 'test':
        test_model(cfg)
    else:
        raise NotImplementedError(f"Action '{cfg.action}' is not implemented")


if __name__ == '__main__':
    main()
