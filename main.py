import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pass  # not implemented yet


if __name__ == '__main__':
    main()
