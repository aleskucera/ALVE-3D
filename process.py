#!/usr/bin/env python
import logging

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.io import set_paths
from src.kitti360 import KITTI360Converter
from src.semantickitti import SemanticKITTIConverter
from src.process import create_superpoints, compute_redal_features

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """ Functions to process the dataset. The options are:
        - convert_dataset: Convert the dataset to the desired format
        - create_superpoints: Create the superpoints for the dataset
        - compute_redal_features: Compute the redal features for the dataset
    """

    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    if cfg.option == 'convert_dataset':
        if cfg.ds.name == 'KITTI-360':
            converter = KITTI360Converter(cfg)
        elif cfg.ds.name == 'SemanticKITTI':
            converter = SemanticKITTIConverter(cfg)
        else:
            raise NotImplementedError(f'The dataset "{cfg.ds.name}" is not supported')
        converter.convert()
    elif cfg.option == 'create_superpoints':
        create_superpoints(cfg)
    elif cfg.option == 'compute_redal_features':
        compute_redal_features(cfg)
    else:
        raise ValueError(f'Option "{cfg.option}" is not supported')


if __name__ == '__main__':
    main()
